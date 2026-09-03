"""训练：Orbit Wars 4P（四人自由混战 FFA）实战方案，从随机权重开始的 PPO 自我对弈。

这是可提交 Kaggle 的实战方案的训练阶段，动作契约与提交端一致：每步取己方驻军最多的
最多 4 颗星球作来源，在"来源×目标"的组合里选一个、或显式选择等待(hold)，选中组合则
全军出击。全局特征专门为三个对手各留了汇总槽。整体链路与 2P 版相同（Config → 特征 →
FourPlayerPolicy → Agent.act → 官方环境 → GAE → PPO → 主循环），差异集中在多来源、hold
分支、以及 16 维全局特征。"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


BOARD_SIZE = 100.0            # 地图边长
BOARD_CENTER = (50.0, 50.0)   # 太阳/地图中心
MAX_SHIPS = 500.0             # 驻军归一化上限
MAX_PROD = 20.0              # 生产值归一化上限
SELF_DIM = 14    # 每颗星球的基础特征维度（与 2P 一致）
CAND_DIM = 18    # 候选特征 = 14 基础 + 4 几何量
GLOBAL_DIM = 16  # 全局特征：步数进度 + 4 名玩家各 3 项(星球数/驻军/生产) + 3 个对手存活标志
MAX_SOURCES = 4  # 每步最多取 4 颗来源
MAX_DISTANCE = math.sqrt(2.0) * BOARD_SIZE  # 对角线，距离归一化用
# 提交端权重的落点：训练刷新 best 时把权重导出到这里，submission_4p/ 便成为可直接打包上传的完整目录
SUBMISSION_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "submission_4p" / "model_weights.npz"


@dataclass
class Config:
    """训练超参数集合（字段含义与 2P 版一致）。"""
    output_dir: Path
    seed: int = 20260711
    games_per_iter: int = 4        # 4P 每局更慢，每迭代局数比 2P 少
    total_iters: int = 2_000
    ppo_epochs: int = 4
    batch_size: int = 128
    hidden_size: int = 256
    num_blocks: int = 2
    dropout: float = 0.05
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    opponent_pool_size: int = 4
    opponent_update_freq: int = 25
    eval_interval: int = 25
    eval_games: int = 24
    episode_steps: int = 500


def set_seed(seed: int) -> None:
    # 固定各随机源并关掉 cudnn 非确定性优化，尽量可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResidualBlock(nn.Module):
    """残差块：LayerNorm → Linear → ReLU → Linear，输出加回输入（与提交端 _encoder 对应）。"""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(self.norm(x)))))
        return residual + self.dropout(x)


class Encoder(nn.Module):
    """编码器：输入投影 + num_blocks 个残差块。来源/候选/全局各用一个。"""

    def __init__(self, input_dim: int, hidden_size: int, num_blocks: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input_proj(x))


class FourPlayerPolicy(nn.Module):
    """玩家相对视角的共享 Actor-Critic：对"来源×目标"组合用 pair_head 打分，另有 hold_head
    给"等待"打分；value_head 估计局面价值。pair 与 hold 的分数共用同一个温度 logit_scale。"""

    def __init__(self, hidden_size: int, num_blocks: int, dropout: float) -> None:
        super().__init__()
        self.source_encoder = Encoder(SELF_DIM, hidden_size, num_blocks, dropout)
        self.target_encoder = Encoder(CAND_DIM, hidden_size, num_blocks, dropout)
        self.global_encoder = Encoder(GLOBAL_DIM, hidden_size, num_blocks, dropout)
        joint_size = 3 * hidden_size
        self.pair_head = nn.Sequential(
            nn.Linear(joint_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1),
        )
        # hold_head 输入 = 各来源隐向量均值 + 全局隐向量，输出 1 个"等待"分数
        self.hold_head = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.value_head = nn.Sequential(
            nn.Linear(joint_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1)
        )
        # 可学习的全局温度，pair 和 hold 分数都乘它；贪心 argmax 下正数缩放不改变谁最大。
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # 线性层正交初始化(gain=√2)、偏置置零。
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(self, sources: torch.Tensor, candidates: torch.Tensor, global_features: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, source_count, target_count, _ = candidates.shape
        source_h = self.source_encoder(sources)
        candidate_h = self.target_encoder(candidates)
        global_h = self.global_encoder(global_features)
        # 把"来源隐向量"和"全局隐向量"广播到每个(来源,目标)格子，与候选隐向量拼接后送 pair_head
        source_pair_h = source_h.unsqueeze(2).expand(-1, -1, target_count, -1)
        global_pair_h = global_h[:, None, None, :].expand(-1, source_count, target_count, -1)
        pair_h = torch.cat([source_pair_h, global_pair_h, candidate_h], dim=-1)
        scale = self.logit_scale.clamp(0.1, 10.0)
        pair_logits = self.pair_head(pair_h).squeeze(-1) * scale
        pair_logits = pair_logits.masked_fill(~pair_mask, float("-inf"))   # 屏蔽非法组合
        hold_logits = self.hold_head(torch.cat([source_h.mean(dim=1), global_h], dim=-1)) * scale
        # 所有 pair 分数拉平后接上 hold 分数，构成一个统一的分类分布（最后一个动作=等待）
        logits = torch.cat([pair_logits.flatten(start_dim=1), hold_logits], dim=1)
        value_h = torch.cat([source_h.mean(dim=1), candidate_h.mean(dim=(1, 2)), global_h], dim=-1)
        return logits, self.value_head(value_h).squeeze(-1)


def fleet_speed(ships: float) -> float:
    # 兵越多飞得越快、封顶 6 倍；与提交端一致。
    ratio = min(math.log(max(ships, 1.0)) / math.log(1000.0), 1.0)
    return 1.0 + 5.0 * ratio**1.5


def body_features(planets: np.ndarray, player_id: int) -> np.ndarray:
    """把每颗星球编码成 SELF_DIM(=14) 维基础特征；含义与 2P / 提交端逐位一致。
    planets 每行 = [id, owner, x, y, radius, ships, production]。"""
    alive = planets[:, 0] >= 0
    owner = planets[:, 1]
    x, y, radius, ships, production = (planets[:, i] for i in range(2, 7))
    dx, dy = x - BOARD_CENTER[0], y - BOARD_CENTER[1]
    center_distance = np.sqrt(dx**2 + dy**2)
    orbit_radius = center_distance + radius
    orbiting = (orbit_radius < 50.0) & (center_distance > 0.5)   # 近太阳绕行判定
    features = np.zeros((len(planets), SELF_DIM), dtype=np.float32)
    features[:, 0:2] = np.stack([x / BOARD_SIZE, y / BOARD_SIZE], axis=1)   # 0-1: 归一化坐标
    features[:, 2] = radius / 5.0                                           # 2: 归一化半径
    features[:, 3] = np.clip(ships, 0.0, MAX_SHIPS) / MAX_SHIPS             # 3: 归一化驻军
    features[:, 4] = np.clip(production, 0.0, MAX_PROD) / MAX_PROD          # 4: 归一化生产值
    features[:, 5] = center_distance / 70.7                                 # 5: 离中心距离
    features[:, 6] = orbiting.astype(np.float32)                           # 6: 是否绕轨道
    features[:, 7:9] = np.stack([dx / BOARD_SIZE, dy / BOARD_SIZE], axis=1) # 7-8: 相对中心方向
    features[:, 9] = ((owner == player_id) & alive).astype(np.float32)                    # 9: 我方
    features[:, 10] = ((owner >= 0) & (owner != player_id) & alive).astype(np.float32)    # 10: 敌方
    features[:, 11] = ((owner == -1) & alive).astype(np.float32)                          # 11: 中立
    features[:, 12] = np.where(orbiting, orbit_radius / BOARD_SIZE, 0.0)                  # 12: 绕行轨道半径
    features[:, 13] = alive.astype(np.float32)                                            # 13: 存活标志
    return features


def extract_state(observation: dict[str, Any], player_id: int) -> dict[str, Any] | None:
    """承接 body_features：把一帧原始观测整理成 FourPlayerPolicy 三个编码器要的三路输入。

    观测里的 planets 是一张原始数值表（每行 [id, owner, x, y, radius, ships, production]，
    坐标、兵数、生产值量纲各异，网络不能直接吃）。原始值 → 网络输入分两层：
      1) body_features 先把每颗星球那一行独立"归一化"成 14 维——各列各除以一个上限压到约
         0~1，所有者拆成我/敌/中立三个 0-1 标志。这步与"从哪出发"无关，对全场星球一次算好。
      2) 本函数在这张归一化表上按 4P 动作契约取行、再补上"来源→目标"的相对量，拼成网络要的三路：
         · source_features = 最多 4 颗来源星球各自的 14 维行（"我能从哪几处出发"）
         · candidates      = [来源数, 目标数, 18] 网格：每个(来源,目标)对 = 目标 14 维 + 4 维几何量
           (方向 dx/dy、距离、粗略 ETA)；这 4 维依赖来源，body_features 算不出，选定来源后现算
         · global_features = 16 维全场摘要（步数进度 + 自己与三对手的星球数/驻军/生产 + 对手存活标志）

    另外把还原动作要用的原始量（来源/目标索引、各对方向角、ships、原始 planets）一并返回。
    返回 None 表示没有可发兵的来源。与提交端 _choose_action 前半段逐位对应。"""
    planet_list = observation.get("planets", [])
    if not planet_list:
        return None
    planets = np.asarray(planet_list, dtype=np.float32)
    alive = planets[:, 0] >= 0
    ships = planets[:, 5]
    owned = alive & (planets[:, 1] == player_id) & (ships >= 4.0)
    source_indices = np.flatnonzero(owned)
    if len(source_indices) == 0:
        return None
    source_indices = source_indices[np.argsort(ships[source_indices])[-MAX_SOURCES:]][::-1]   # 取驻军最多的 4 颗
    target_indices = np.flatnonzero(alive)
    if len(target_indices) == 0:
        return None

    # 候选是 [来源数, 目标数, 18] 的网格：每个(来源,目标)对拼 14 维目标特征 + 4 维几何量
    features = body_features(planets, player_id)   # 上一步：全场星球的归一化行表；下面网络只吃这张表，不再碰原始 planets
    source_features = features[source_indices]      # 取出来源那几行的归一化特征
    source_xy = planets[source_indices, 2:4]        # 原始坐标仅在此用于算相对几何
    target_xy = planets[target_indices, 2:4]
    dx = target_xy[None, :, 0] - source_xy[:, None, 0]   # 广播成 [来源,目标] 原始位移矩阵（还没归一化）
    dy = target_xy[None, :, 1] - source_xy[:, None, 1]
    distance = np.sqrt(dx**2 + dy**2)
    candidates = np.zeros((len(source_indices), len(target_indices), CAND_DIM), dtype=np.float32)
    candidates[:, :, :SELF_DIM] = features[target_indices][None, :, :]   # 前 14 列：把候选目标的归一化行广播到每个来源
    candidates[:, :, 14] = dx / BOARD_SIZE                # 14-15: 指向目标的方向（÷边长压到约 -1~1）
    candidates[:, :, 15] = dy / BOARD_SIZE
    candidates[:, :, 16] = distance / MAX_DISTANCE        # 16: 到目标的距离（÷对角线压到 0~1）
    speeds = np.array([fleet_speed(float(ships[index])) for index in source_indices], dtype=np.float32)
    candidates[:, :, 17] = np.clip(distance / speeds[:, None], 0.0, 50.0) / 50.0   # 17: 粗略 ETA=距离÷速度，再压到 0~1
    pair_mask = (distance > 1e-6) & (target_indices[None, :] != source_indices[:, None])   # 合法组合（排除自己→自己）

    # 16 维全局：步数进度(1) + 自己与三个对手各 3 项(星球数/驻军/生产, 共 12) + 三个对手存活标志(3)；各项同样÷常数压到约 0~1
    global_values: list[float] = [observation.get("step", 0) / 500.0]   # 维度 0: 步数进度（当前步/最大步）
    for pid in [player_id] + [rival for rival in range(4) if rival != player_id]:   # 先自己，再三个对手
        owned_by_pid = alive & (planets[:, 1] == pid)
        global_values.extend([owned_by_pid.sum() / 64.0, np.clip(planets[owned_by_pid, 5].sum(), 0.0, 32000.0) / 32000.0, np.clip(planets[owned_by_pid, 6].sum(), 0.0, 1280.0) / 1280.0])   # 该玩家的[星球数, 总驻军, 总生产值]
    # 最后三项：每个对手是否还拥有星球（是否还"活着"）
    for rival in [pid for pid in range(4) if pid != player_id]:
        global_values.append(float(np.any(alive & (planets[:, 1] == rival))))   # 该对手存活标志（1=还有星球）
    return {
        "source_features": source_features,                     # 最多 4 颗来源星球各自的 14 维特征（网络输入）
        "candidates": candidates,                               # [来源数, 目标数, 18] 候选网格特征（网络输入）
        "global_features": np.asarray(global_values, dtype=np.float32),  # 16 维全局特征（网络输入）
        "pair_mask": pair_mask,                                 # 合法(来源,目标)组合掩码（True=可发兵）
        "source_indices": source_indices,                       # 各来源星球在 planets 里的行号
        "target_indices": target_indices,                       # 各候选目标在 planets 里的行号
        "angles": np.arctan2(dy, dx),                           # 每个(来源,目标)对的方向角（弧度，还原动作用）
        "ships": ships,                                         # 全场星球驻军数组（取来源发兵数用）
        "planets": planets,                                     # 原始星球表（取来源星球 id 用）
    }


class FourPlayerRunner:
    """封装官方环境的一局 4P 对战：规则由官方环境执行，这里只推进状态、读观测与奖励。"""

    def __init__(self, seed: int, episode_steps: int) -> None:
        try:
            from kaggle_environments import make
        except ImportError as exc:
            raise RuntimeError("Missing kaggle_environments. Install the pinned Ubuntu requirements before training.") from exc
        self.env = make("orbit_wars", configuration={"seed": seed, "randomSeed": seed, "episodeSteps": episode_steps}, debug=True)
        self.env.reset(num_agents=4)
        self.states = self.env.step([[], [], [], []])   # 先走一步空动作进入活动态
        self.done = False

    def step(self, actions: list[list[list[float]]]) -> None:
        self.states = self.env.step(actions)
        self.done = all(state.get("status", "ACTIVE") != "ACTIVE" for state in self.states)

    @property
    def rewards(self) -> list[float]:
        return [float(state.get("reward", 0.0) or 0.0) for state in self.states]


class Agent:
    """把网络输出转成官方动作。deterministic 决定贪心还是采样；返回 (动作, 经验)。
    动作索引落在 [0, 来源数×目标数) 表示某个发兵组合，等于最后一个索引则表示 hold（等待）。"""

    def __init__(self, model: FourPlayerPolicy, device: torch.device) -> None:
        self.model, self.device = model, device

    @torch.no_grad()
    def act(self, observation: dict[str, Any], player_id: int, deterministic: bool) -> tuple[list[list[float]], dict | None]:
        state = extract_state(observation, player_id)
        if state is None or not state["pair_mask"].any():
            return [], None
        source = torch.from_numpy(state["source_features"]).unsqueeze(0).float().to(self.device)
        candidates = torch.from_numpy(state["candidates"]).unsqueeze(0).float().to(self.device)
        global_features = torch.from_numpy(state["global_features"]).unsqueeze(0).float().to(self.device)
        pair_mask = torch.from_numpy(state["pair_mask"]).unsqueeze(0).bool().to(self.device)
        logits, value = self.model(source, candidates, global_features, pair_mask)
        distribution = Categorical(logits=logits.squeeze(0))
        action_index = logits.argmax(dim=1).item() if deterministic else distribution.sample().item()
        source_count, target_count = state["pair_mask"].shape
        hold_index = source_count * target_count   # 拉平后紧跟在所有 pair 之后的那个索引 = hold
        actions: list[list[float]] = []
        if action_index != hold_index:             # 非 hold：还原成(来源槽,目标槽)并全军出击
            source_slot, target_slot = divmod(action_index, target_count)
            if state["pair_mask"][source_slot, target_slot]:
                source_index = state["source_indices"][source_slot]
                source_id = int(state["planets"][source_index, 0])
                actions = [[source_id, float(state["angles"][source_slot, target_slot]), int(state["ships"][source_index])]]
        experience = {
            "source_features": state["source_features"].copy(),
            "candidates": state["candidates"].copy(),
            "global_features": state["global_features"].copy(),
            "pair_mask": state["pair_mask"].copy(),
            "action": action_index,
            "log_prob": distribution.log_prob(torch.tensor(action_index, device=self.device)).item(),
            "value": value.item(),
        }
        return actions, experience


class Buffer:
    """经验缓冲：按时间顺序攒 (经验, 奖励, 是否终局)，一个迭代结束后统一 PPO 更新。"""

    def __init__(self) -> None:
        self.experiences: list[dict] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def add(self, experience: dict | None, reward: float, done: bool) -> None:
        if experience is not None:
            self.experiences.append(experience)
            self.rewards.append(float(reward))
            self.dones.append(done)

    def end_episode(self, terminal_reward: float) -> None:
        # 末步塑形奖励换成环境真实胜负奖励，并标记终局
        if self.rewards:
            self.rewards[-1] = float(terminal_reward)
            self.dones[-1] = True

    def extend(self, other: "Buffer") -> None:
        self.experiences.extend(other.experiences)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)

    def clear(self) -> None:
        self.experiences.clear(); self.rewards.clear(); self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


def compute_gae(rewards: list[float], values: list[float], dones: list[bool], cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """广义优势估计(GAE)：从后往前按 γλ 衰减累加 TD 误差成优势，returns=优势+价值 作回归目标。
    终局处 nonterminal=0 切断向未来的传播。与 2P 版实现一致。"""
    returns, advantages = np.zeros(len(rewards), np.float32), np.zeros(len(rewards), np.float32)
    gae, next_value = 0.0, 0.0   # gae=上一步(更靠后)的优势 A_{t+1}；next_value=下一状态价值 V(s_{t+1})，末步之后取 0
    for index in reversed(range(len(rewards))):   # 必须倒序：A_t 依赖已算好的 A_{t+1}
        nonterminal = 1.0 - float(dones[index])   # 终局→0（切断未来），非终局→1
        # 单步 TD 误差 δ_t = r_t + γ·V(s_{t+1})·nonterminal − V(s_t)
        delta = rewards[index] + cfg.gamma * next_value * nonterminal - values[index]
        # 递推优势 A_t = δ_t + γλ·nonterminal·A_{t+1}：把后续优势按 γλ 指数衰减累加进来
        gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
        advantages[index] = gae
        returns[index] = gae + values[index]      # 回归目标 return_t = A_t + V(s_t)，供价值头拟合
        next_value = values[index]                # 本步的 V(s_t) 即上一步（更靠前）的 V(s_{t+1})
    return returns, advantages


def ppo_update(model: FourPlayerPolicy, optimizer: torch.optim.Optimizer, buffer: Buffer, device: torch.device, cfg: Config) -> dict[str, float]:
    """PPO 更新。各步来源数/目标数不同，需补零对齐成规整张量；补零会改变拉平后 hold 的位置，
    所以下面要把每步的动作索引重映射到"本批统一形状"下的正确位置。"""
    if not buffer:
        return {"policy": 0.0, "value": 0.0, "entropy": 0.0}
    # ── ① 用整批经验算 GAE：advantages 供策略损失加权，returns 作价值头回归目标 ──
    returns, advantages = compute_gae(buffer.rewards, [item["value"] for item in buffer.experiences], buffer.dones, cfg)

    # ── ② 变长的来源×目标网格补零对齐：取本批最大来源/目标数，把每步补零到同一形状才能堆成规整批张量 ──
    count = len(buffer)
    max_sources = max(item["source_features"].shape[0] for item in buffer.experiences)   # 本批最大来源数
    max_targets = max(item["candidates"].shape[1] for item in buffer.experiences)        # 本批最大目标数
    sources = np.zeros((count, max_sources, SELF_DIM), np.float32)                # 来源特征
    candidates = np.zeros((count, max_sources, max_targets, CAND_DIM), np.float32) # 候选网格（不足处留 0）
    globals_ = np.zeros((count, GLOBAL_DIM), np.float32)                          # 全局特征
    masks = np.zeros((count, max_sources, max_targets), bool)                     # 合法(来源,目标)掩码（补零处 False）
    actions, old_log_probs = np.zeros(count, np.int64), np.zeros(count, np.float32)  # 当时所选动作索引 / 旧策略 log π
    for index, item in enumerate(buffer.experiences):                            # 逐步把真实长度数据填进补零张量
        source_count, target_count = item["pair_mask"].shape
        sources[index, :source_count] = item["source_features"]
        candidates[index, :source_count, :target_count] = item["candidates"]
        globals_[index] = item["global_features"]
        masks[index, :source_count, :target_count] = item["pair_mask"]
        # ★ 关键：补零把每行目标数从 target_count 撑到 max_targets，展平后动作索引整体错位，必须重映射：
        #   · hold 动作要挪到本批统一的末尾索引 max_sources*max_targets；
        #   · pair 动作要按新行宽 max_targets 重算展平下标（原来按 target_count 展平）。
        action = item["action"]
        actions[index] = max_sources * max_targets if action == source_count * target_count else (action // target_count) * max_targets + action % target_count
        old_log_probs[index] = item["log_prob"]

    # ── ③ 一次性搬上目标设备，并标准化优势（均值0/方差1，稳定训练） ──
    source_t, candidate_t, global_t, mask_t, action_t, old_lp_t, return_t, advantage_t = [torch.from_numpy(array).to(device) for array in (sources, candidates, globals_, masks, actions, old_log_probs, returns, advantages)]
    advantage_t = (advantage_t.float() - advantage_t.float().mean()) / (advantage_t.float().std(unbiased=False) + 1e-8)   # 优势标准化

    # ── ④ 多轮小批量 PPO 优化：同一批数据反复用 ppo_epochs 轮 ──
    order, totals, updates = np.arange(count), np.zeros(3, np.float64), 0   # totals 累计 policy/value/entropy，末尾取均值
    model.train()
    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(order)           # 每轮打乱样本顺序再切小批
        for start in range(0, count, cfg.batch_size):
            indices = torch.as_tensor(order[start:start + cfg.batch_size], device=device)
            # 用当前（新）策略重新前向这批样本，拿到新 logits 与价值预测
            logits, values = model(source_t[indices].float(), candidate_t[indices].float(), global_t[indices].float(), mask_t[indices].bool())
            distribution = Categorical(logits=logits)
            log_prob = distribution.log_prob(action_t[indices].long())   # 新策略对"当时所选动作"的 log π
            entropy = distribution.entropy().mean()                      # 策略熵，越大越随机（鼓励探索）
            ratio = (log_prob - old_lp_t[indices].float()).exp()   # 新旧策略概率比 r = exp(logπ_new − logπ_old)
            # 策略损失（PPO 截断目标）：把 r 夹在 [1-ε, 1+ε] 内取较小者，限制单次更新步幅；前置负号=最大化目标转最小化损失
            policy_loss = -torch.minimum(ratio * advantage_t[indices], ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantage_t[indices]).mean()
            value_loss = F.mse_loss(values, return_t[indices].float())   # 价值损失：价值头预测 vs GAE 回报的 MSE
            # 总损失 = 策略损失 + value_coef·价值损失 − entropy_coef·熵（熵项取负 = 鼓励探索）
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm); optimizer.step()   # 反传+梯度裁剪+更新
            totals += (policy_loss.item(), value_loss.item(), entropy.item()); updates += 1
    model.eval()
    return dict(zip(("policy", "value", "entropy"), totals / max(updates, 1)))   # 返回三项损失均值供日志


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    # 深拷权重到 CPU：给对手池/最佳状态存档用。
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def shaped_reward(observation: dict[str, Any], player_id: int) -> float:
    """塑形奖励（4P）：局面分 = 兵力差 + 星球数差，其中"对手项"取三个对手的均值。
    兵力 = 星球驻军 + 在途舰队，与官方计分口径一致。终局那步会被真实胜负奖励覆盖。"""
    planets = np.asarray(observation.get("planets", []), dtype=np.float32)
    if len(planets) == 0:
        return 0.0
    # fleets 每行 = [id, owner, x, y, angle, from_planet_id, ships]，owner 第 1 列、ships 第 6 列
    fleets = np.asarray(observation.get("fleets", []), dtype=np.float32)
    alive = planets[:, 0] >= 0

    def total_ships(pid: int) -> float:
        # 某玩家总兵力 = 其所有存活星球驻军 + 其在途舰队兵力
        garrisons = planets[alive & (planets[:, 1] == pid), 5].sum()
        in_flight = fleets[fleets[:, 1] == pid, 6].sum() if len(fleets) else 0.0
        return float(garrisons + in_flight)

    rival_ids = [rival for rival in range(4) if rival != player_id]
    own_planets = (alive & (planets[:, 1] == player_id)).sum()
    rival_planets = np.mean([(alive & (planets[:, 1] == rival)).sum() for rival in rival_ids])   # 对手星球数均值
    ship_gap = (total_ships(player_id) - np.mean([total_ships(rival) for rival in rival_ids])) / 2000.0   # 与对手均值的兵力差
    return float(ship_gap + (own_planets - rival_planets) / 100.0)


def evaluate(model: FourPlayerPolicy, reference_state: dict[str, torch.Tensor], device: torch.device, cfg: Config, seeds: list[int]) -> float:
    """让当前模型占 1 个座位、参考(best)模型占其余 3 个座位贪心对打，返回当前模型胜率。轮换座位。"""
    reference = FourPlayerPolicy(cfg.hidden_size, cfg.num_blocks, cfg.dropout).to(device)
    reference.load_state_dict(reference_state)
    reference.eval()
    learner, opponent = Agent(model, device), Agent(reference, device)
    wins = 0
    for game_number, seed in enumerate(seeds):
        learner_id = game_number % 4   # 轮流坐 0~3 号位
        game = FourPlayerRunner(seed, cfg.episode_steps)
        while not game.done:
            actions = []
            for player_id, state in enumerate(game.states):
                observation = state.get("observation", {})
                agent = learner if player_id == learner_id else opponent
                action, _ = agent.act(observation, player_id, deterministic=True)
                actions.append(action)
            game.step(actions)
        wins += int(game.rewards[learner_id] > 0)
    return wins / len(seeds)


def save_checkpoint(path: Path, model: FourPlayerPolicy, optimizer: torch.optim.Optimizer, iteration: int, metric: float, cfg: Config) -> None:
    # 存 .pt 检查点，用于存档与续训。format / architecture / action_contract 都是自描述元数据
    # （记录这份检查点出自哪个赛制、什么结构、什么动作契约），供人查阅，代码不读它们。
    torch.save(
        {
            "format": "orbit-wars-4p-training-checkpoint-v1",
            "iteration": iteration,
            "metric": metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": {"hidden_size": cfg.hidden_size, "num_blocks": cfg.num_blocks, "dropout": cfg.dropout},
            "action_contract": "up-to-four richest sources x all targets + hold; one all-in launch at most",
        },
        path,
    )


def export_submission_weights(model: FourPlayerPolicy, path: Path) -> Path:
    """把当前权重导出成提交端能直接读的 model_weights.npz。

    提交端不 import torch、只认 .npz，所以 .pt 不能直接当提交权重：这里把 state_dict 里的
    每个张量转成 NumPy 数组存进 npz，键名保持模块路径（如 source_encoder.input_proj.weight）
    原样不动——提交端正是按这些键名取参数的。
    """
    path.parent.mkdir(parents=True, exist_ok=True)   # 单独把训练脚本放到别处跑时，目标目录可能不存在
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    np.savez(path, **arrays)
    return path


def train(cfg: Config) -> None:
    """训练主循环：从随机权重起，每迭代用历史对手池自我对弈（学习者占 1 座、对手占其余 3 座）
    收集经验 → PPO 更新 → 定期评估存档、定期把当前权重存入对手池。"""
    # ── ① 准备：固定随机种子、选设备、建存档目录并把本次超参落盘 ──
    set_seed(cfg.seed)   # 固定 Python/NumPy/Torch 随机源，尽量可复现
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 有 GPU 用 GPU，否则退回 CPU
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()}, handle, indent=2)   # 存一份超参快照便于复盘

    # ── ② 建模型与优化器：从随机初始化起（无预训练权重） ──
    model = FourPlayerPolicy(cfg.hidden_size, cfg.num_blocks, cfg.dropout).to(device)
    model.eval()   # 采样对局时用 eval（关 dropout）；只有 ppo_update 内部才临时切回 train
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ── ③ 自我对弈记账：对手池 + best 快照 + 经验缓冲 ──
    # 对手池：初始只放随机权重的自己，训练中不断加入新版本、每局随机抽一个当参考对手
    pool: deque[dict[str, torch.Tensor]] = deque([clone_state(model)], maxlen=cfg.opponent_pool_size)
    best_state, best_win_rate = clone_state(model), 0.0   # best_state=最强权重快照，作评估参照对手
    rng, buffer = np.random.default_rng(cfg.seed), Buffer()   # rng=抽对手/开局种子的独立随机源；buffer 跨整迭代累计经验
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} output={cfg.output_dir}")

    for iteration in range(1, cfg.total_iters + 1):
        started, terminal_rewards = time.perf_counter(), []

        # ── ④ 自我对弈：本迭代打 games_per_iter 局，学习者占 1 座、抽到的历史对手占其余 3 座 ──
        for game_number in range(cfg.games_per_iter):
            opponent_model = FourPlayerPolicy(cfg.hidden_size, cfg.num_blocks, cfg.dropout).to(device)
            opponent_model.load_state_dict(pool[int(rng.integers(len(pool)))])   # 抽历史版本当对手（权重固定、不训练）
            opponent_model.eval()
            learner, opponent = Agent(model, device), Agent(opponent_model, device)   # learner=在训模型；其余 3 座共用同一个 opponent
            learner_id = game_number % 4   # 轮换学习者座位（0~3 交替），抵消座位偏差
            game, episode = FourPlayerRunner(int(rng.integers(2**31 - 1)), cfg.episode_steps), Buffer()   # 随机开局种子；episode=单局经验
            while not game.done:               # 一局到底：每步四座各决策 → 推进环境 → 记学习者的经验
                actions, experience = [], None
                # 四个座位都要出动作；只有学习者座位的经验被收集用于训练
                for player_id, state in enumerate(game.states):
                    agent = learner if player_id == learner_id else opponent   # 学习者座位用在训模型，其余用历史对手
                    action, exp = agent.act(state.get("observation", {}), player_id, deterministic=False)   # 采样保留探索
                    actions.append(action)
                    if player_id == learner_id:
                        experience = exp   # 只留学习者那座的经验
                game.step(actions)   # 官方环境同时结算四座动作
                learner_observation = game.states[learner_id].get("observation", {})
                episode.add(experience, shaped_reward(learner_observation, learner_id), game.done)   # 用动作后的局面打塑形奖励；等待步 experience=None 不记
            episode.end_episode(game.rewards[learner_id])   # 末步塑形奖励换成官方真实名次奖励
            terminal_rewards.append(game.rewards[learner_id]); buffer.extend(episode)

        # ── ⑤ PPO 更新：用本迭代经验更新一次，清空 buffer，并定期把当前权重存进对手池 ──
        stats, sample_count = ppo_update(model, optimizer, buffer, device, cfg), len(buffer)
        buffer.clear()   # 经验只用一轮（on-policy），用完即弃
        if iteration % cfg.opponent_update_freq == 0:   # 定期把当前权重快照进对手池，让对手也变强
            pool.append(clone_state(model))

        # ── ⑥ 定期评估与存档：贪心对打 best，胜率达标才刷新 best 并导出提交权重 ──
        if iteration % cfg.eval_interval == 0:
            seeds = [cfg.seed + iteration * 10_000 + offset for offset in range(cfg.eval_games)]   # 固定种子，评估可比
            win_rate = evaluate(model, best_state, device, cfg, seeds)   # 当前模型 vs best 的胜率
            save_checkpoint(cfg.output_dir / "latest.pt", model, optimizer, iteration, win_rate, cfg)   # 每次评估都存 latest（可续训）
            if win_rate >= 0.45:  # 四人局随机基线约 25%，要明显高于它才替换 best
                best_state, best_win_rate = clone_state(model), win_rate
                save_checkpoint(cfg.output_dir / "best.pt", model, optimizer, iteration, win_rate, cfg)
                # 只在刷新 best 时导出，submission_4p/ 里的权重因此始终等于当前最好的模型
                export_submission_weights(model, SUBMISSION_WEIGHTS_PATH)
            print(f"iter={iteration:5d} samples={sample_count:5d} terminal_mean={np.mean(terminal_rewards):+.3f} eval_vs_best={win_rate:.3f} best={best_win_rate:.3f}")
        elif iteration % 5 == 0:   # 非评估迭代每 5 次打印一次损失，便于盯训练是否健康
            print(f"iter={iteration:5d} samples={sample_count:5d} terminal_mean={np.mean(terminal_rewards):+.3f} policy={stats['policy']:.4f} value={stats['value']:.4f} entropy={stats['entropy']:.4f} sec={time.perf_counter() - started:.1f}")
    save_checkpoint(cfg.output_dir / "final.pt", model, optimizer, cfg.total_iters, best_win_rate, cfg)   # 训练结束存最终检查点


def parse_args() -> Config:
    # 命令行只暴露常调超参，用命名参数构造 Config 避免字段错位。
    parser = argparse.ArgumentParser(description="Train a four-player Orbit Wars PPO policy from random initialization.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/orbit_wars_4p"))
    parser.add_argument("--total-iters", type=int, default=2_000)
    parser.add_argument("--games-per-iter", type=int, default=4)
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--hidden-size", type=int, default=256)
    arguments = parser.parse_args()
    return Config(
        output_dir=arguments.output_dir,
        total_iters=arguments.total_iters,
        games_per_iter=arguments.games_per_iter,
        eval_games=arguments.eval_games,
        seed=arguments.seed,
        hidden_size=arguments.hidden_size,
    )


if __name__ == "__main__":
    train(parse_args())
