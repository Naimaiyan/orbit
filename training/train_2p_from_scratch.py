"""训练：Orbit Wars 2P（1v1）实战方案，从随机权重开始的 PPO 自我对弈。

这是可提交 Kaggle 的实战方案的训练阶段。动作契约与提交端严格一致：每步只从"己方
驻军最多的一颗"星球出发，为它在其余星球里选 1 个目标，然后全军出击——训练怎么决策，
线上就怎么决策。整体链路：Config(超参) → planet_features(观测→特征) → TargetPolicy
(Actor-Critic 网络) → Agent.act(采样出动作) → GameRunner(调官方环境跑对局) →
RolloutBuffer + shaped_reward(收集经验与奖励) → compute_gae → ppo_update → train(主循环)。"""

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
MAX_DISTANCE = math.sqrt(2.0) * BOARD_SIZE  # 对角线，距离归一化用
SELF_DIM = 14    # 来源星球（及每颗星球）的基础特征维度
CAND_DIM = 18    # 候选目标特征维度 = 14 基础 + 4 几何量(方向 dx/dy、距离、粗略 ETA)
GLOBAL_DIM = 8   # 全局特征维度（步数进度 + 双方星球数/总驻军/总生产）
# 提交端权重的落点：训练刷新 best 时把权重导出到这里，submission_2p/ 便成为可直接打包上传的完整目录
SUBMISSION_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "submission_2p" / "model_weights.npz"


@dataclass
class Config:
    """训练超参数集合。"""
    output_dir: Path               # 权重/日志输出目录
    seed: int = 20260630           # 随机种子（复现用）
    games_per_iter: int = 8        # 每个迭代自我对弈多少局
    total_iters: int = 2_000       # 总迭代数
    ppo_epochs: int = 4            # 每批数据重复优化几轮
    batch_size: int = 256
    hidden_size: int = 256         # 编码器/隐层宽度
    num_blocks: int = 2            # 每个编码器的残差块数
    dropout: float = 0.05
    lr: float = 3e-4
    gamma: float = 0.99            # 折扣因子
    gae_lambda: float = 0.95       # GAE 的 λ
    clip_eps: float = 0.2          # PPO 截断范围
    value_coef: float = 0.5        # 价值损失权重
    entropy_coef: float = 0.01     # 熵奖励权重（鼓励探索）
    max_grad_norm: float = 0.5     # 梯度裁剪上限
    opponent_pool_size: int = 4    # 历史对手池容量
    opponent_update_freq: int = 25 # 每隔多少迭代把当前权重存入对手池
    eval_interval: int = 25        # 每隔多少迭代评估一次并存档
    eval_games: int = 24           # 每次评估打多少局
    episode_steps: int = 500       # 单局最大步数（官方上限）


def set_seed(seed: int) -> None:
    # 统一固定 Python/NumPy/PyTorch 的随机源，并关掉 cudnn 的非确定性优化，尽量让训练可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResidualBlock(nn.Module):
    """残差块：LayerNorm → Linear → ReLU → Linear，输出再加回输入（提交端 _encoder 手写的就是它）。"""

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
    """编码器：输入投影到 hidden_size，再过 num_blocks 个残差块。三路特征各用一个编码器。"""

    def __init__(self, input_dim: int, hidden_size: int, num_blocks: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input_proj(x))


class TargetPolicy(nn.Module):
    """针对一颗固定来源星球、在所有合法目标上做选择的 Actor-Critic 网络。
    target_head 输出各目标分数（策略），value_head 输出状态价值（评论家）。"""

    def __init__(self, hidden_size: int, num_blocks: int, dropout: float) -> None:
        super().__init__()
        self.self_encoder = Encoder(SELF_DIM, hidden_size, num_blocks, dropout)
        self.candidate_encoder = Encoder(CAND_DIM, hidden_size, num_blocks, dropout)
        self.global_encoder = Encoder(GLOBAL_DIM, hidden_size, num_blocks, dropout)
        joint_size = hidden_size * 3
        self.target_head = nn.Sequential(
            nn.Linear(joint_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(joint_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        # logit_scale 是可学习的全局温度：初值 1 让随机初始化的分类策略保持足够的探索性。
        # 它只影响采样时各动作的相对概率；提交端贪心 argmax 时正数缩放不改变谁最大。
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # 线性层用正交初始化（gain=√2 配合 ReLU），偏置置零——常见的策略网络初始化方式。
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, candidate_count, _ = candidate_features.shape
        # 三路编码，再把"来源+全局"广播到每个候选后与候选隐向量拼接，送 target_head 打分
        self_h = self.self_encoder(self_features)
        candidate_h = self.candidate_encoder(candidate_features)
        global_h = self.global_encoder(global_features)
        joint = torch.cat(
            [
                self_h.unsqueeze(1).expand(-1, candidate_count, -1),
                global_h.unsqueeze(1).expand(-1, candidate_count, -1),
                candidate_h,
            ],
            dim=-1,
        )
        logits = self.target_head(joint).squeeze(-1) * self.logit_scale.clamp(0.1, 10.0)
        logits = logits.masked_fill(~candidate_mask, float("-inf"))   # 非法目标置 -inf
        # 价值头输入用候选隐向量的均值（对目标数不敏感），评估当前局面好坏
        value_input = torch.cat([self_h, global_h, candidate_h.mean(dim=1)], dim=-1)
        values = self.value_head(value_input).squeeze(-1)
        return logits, values


def fleet_speed(ships: float) -> float:
    # 舰队速度随兵数对数增长、封顶 6 倍；与提交端 _fleet_speed 完全一致（算候选 ETA 用）。
    ratio = min(math.log(max(ships, 1.0)) / math.log(1000.0), 1.0)
    return 1.0 + 5.0 * ratio**1.5


def planet_features(planets: np.ndarray, player_id: int) -> np.ndarray:
    """把每颗星球编码成 SELF_DIM(=14) 维基础特征；含义与提交端 _planet_features 逐位一致。
    planets 每行 = [id, owner, x, y, radius, ships, production]。"""
    alive = planets[:, 0] >= 0
    owner = planets[:, 1]
    x, y, radius, ships, production = (planets[:, i] for i in range(2, 7))
    dx = x - BOARD_CENTER[0]
    dy = y - BOARD_CENTER[1]
    distance_to_center = np.sqrt(dx**2 + dy**2)
    orbit_radius = distance_to_center + radius
    is_orbiting = (orbit_radius < 50.0) & (distance_to_center > 0.5)   # 近太阳绕行判定
    features = np.zeros((len(planets), SELF_DIM), dtype=np.float32)
    features[:, 0] = x / BOARD_SIZE                          # 0-1: 归一化坐标
    features[:, 1] = y / BOARD_SIZE
    features[:, 2] = radius / 5.0                            # 2: 归一化半径
    features[:, 3] = np.clip(ships, 0.0, MAX_SHIPS) / MAX_SHIPS       # 3: 归一化驻军
    features[:, 4] = np.clip(production, 0.0, MAX_PROD) / MAX_PROD    # 4: 归一化生产值
    features[:, 5] = distance_to_center / 70.7              # 5: 离中心距离
    features[:, 6] = is_orbiting.astype(np.float32)         # 6: 是否绕轨道
    features[:, 7] = dx / BOARD_SIZE                        # 7-8: 相对中心方向
    features[:, 8] = dy / BOARD_SIZE
    features[:, 9] = ((owner == player_id) & alive).astype(np.float32)                    # 9: 我方
    features[:, 10] = ((owner >= 0) & (owner != player_id) & alive).astype(np.float32)    # 10: 敌方
    features[:, 11] = ((owner == -1) & alive).astype(np.float32)                          # 11: 中立
    features[:, 12] = np.where(is_orbiting, orbit_radius / BOARD_SIZE, 0.0)               # 12: 绕行轨道半径
    features[:, 13] = alive.astype(np.float32)                                            # 13: 存活标志
    return features


def extract_state(observation: dict[str, Any], player_id: int, min_ships: float = 4.0) -> dict[str, Any] | None:
    """承接 planet_features：把一帧原始观测整理成 TargetPolicy 三个编码器要的三路输入。

    观测里的 planets 是一张原始数值表（每行 [id, owner, x, y, radius, ships, production]，
    坐标、兵数、生产值量纲各异，网络不能直接吃）。原始值 → 网络输入分两层：
      1) planet_features 先把每颗星球那一行独立"归一化"成 14 维——各列各除以一个上限压到约
         0~1，所有者拆成我/敌/中立三个 0-1 标志。这步与"从哪出发"无关，对全场星球一次算好。
      2) 本函数在这张归一化表上按动作契约取行、再补上"来源→目标"的相对量，拼成网络要的三路：
         · self_features      = 来源星球那一行 14 维（"我从哪出发"）
         · candidate_features = 每个候选目标 14 维 + 4 维几何量(方向 dx/dy、距离、粗略 ETA)；
           这 4 维依赖来源，planet_features 算不出，只能在选定来源后现算（"打它顺不顺路"）
         · global_features    = 一份 8 维全场摘要（步数进度 + 双方星球数/总驻军/总生产）

    另外把"网络看不到、但把动作还原成官方指令时要用"的原始量（来源/目标索引、源驻军、各目标
    方向角、原始 planets）一并返回。返回 None 表示这步没有合法来源（该等待）。
    与提交端 _choose_action 的前半段逐位对应。"""
    planets_list = observation.get("planets", [])
    if not planets_list:
        return None
    planets = np.asarray(planets_list, dtype=np.float32)
    alive = planets[:, 0] >= 0
    owner = planets[:, 1]
    ships = planets[:, 5]
    mine = (owner == player_id) & alive & (ships >= min_ships)   # 可发兵的我方星球（驻军≥4）
    if not mine.any():
        return None
    source_candidates = np.flatnonzero(mine)
    source_index = source_candidates[np.argmax(ships[source_candidates])]   # 唯一来源=驻军最多那颗
    target_indices = np.flatnonzero(alive)
    target_indices = target_indices[target_indices != source_index]        # 候选=其余存活星球
    if len(target_indices) == 0:
        return None

    # 候选特征 = 14 维目标基础特征 + 4 维"从来源瞄准它"的几何量
    body_features = planet_features(planets, player_id)   # 上一步：全场星球的归一化行表；下面网络只吃这张表，不再碰原始 planets
    source_x, source_y = planets[source_index, 2], planets[source_index, 3]   # 原始坐标仅在此用于算相对几何
    dx = planets[target_indices, 2] - source_x           # 来源→各目标的原始位移（还没归一化）
    dy = planets[target_indices, 3] - source_y
    distance = np.sqrt(dx**2 + dy**2)
    candidate_features = np.zeros((len(target_indices), CAND_DIM), dtype=np.float32)
    candidate_features[:, :SELF_DIM] = body_features[target_indices]   # 前 14 列：直接搬候选目标那几行的归一化特征
    candidate_features[:, 14] = dx / BOARD_SIZE                  # 14-15: 指向目标的方向（÷边长压到约 -1~1）
    candidate_features[:, 15] = dy / BOARD_SIZE
    candidate_features[:, 16] = distance / MAX_DISTANCE          # 16: 到目标的距离（÷对角线压到 0~1）
    candidate_features[:, 17] = np.clip(distance / fleet_speed(float(ships[source_index])), 0.0, 50.0) / 50.0  # 17: 粗略 ETA=距离÷速度，再压到 0~1
    candidate_mask = distance > 1e-6   # 排除与来源重合的目标（自己打自己无意义）

    # 8 维全局特征：步数进度 + 我/敌/中立星球数 + 我/敌总驻军 + 我/敌总生产（每项同样÷一个常数压到约 0~1；与提交端一致）
    enemy = (owner >= 0) & (owner != player_id) & alive
    neutral = (owner == -1) & alive
    global_features = np.array(
        [
            observation.get("step", 0) / 500.0,                             # 0: 步数进度（当前步/最大步）
            mine.sum() / 64.0,                                             # 1: 我方星球数
            enemy.sum() / 64.0,                                            # 2: 敌方星球数
            neutral.sum() / 64.0,                                         # 3: 中立星球数
            np.clip(ships[mine].sum(), 0.0, 32000.0) / 32000.0,           # 4: 我方总驻军
            np.clip(ships[enemy].sum(), 0.0, 32000.0) / 32000.0,          # 5: 敌方总驻军
            np.clip(planets[mine, 6].sum(), 0.0, 1280.0) / 1280.0,        # 6: 我方总生产值
            np.clip(planets[enemy, 6].sum(), 0.0, 1280.0) / 1280.0,       # 7: 敌方总生产值
        ],
        dtype=np.float32,
    )
    # 除网络输入外，一并返回还原动作要用的信息：来源/目标索引、源驻军、各目标方向角、原始 planets
    return {
        "self_features": body_features[source_index],   # 来源星球的 14 维特征（网络输入）
        "candidate_features": candidate_features,       # 各候选目标的 18 维特征（网络输入）
        "global_features": global_features,             # 8 维全局特征（网络输入）
        "candidate_mask": candidate_mask,               # 候选有效性掩码（True=合法目标）
        "source_index": source_index,                   # 来源星球在 planets 里的行号
        "target_indices": target_indices,               # 各候选目标在 planets 里的行号
        "source_ships": int(ships[source_index]),       # 来源驻军数（全军出击时的发兵数）
        "angles": np.arctan2(dy, dx),                   # 来源→各目标的方向角（弧度，还原动作用）
        "planets": planets,                             # 原始星球表（取来源星球 id 用）
    }


class GameRunner:
    """封装官方 kaggle_environments 的一局 2P 对战：游戏规则完全交给官方环境执行。"""

    def __init__(self, seed: int, episode_steps: int) -> None:
        try:
            from kaggle_environments import make
        except ImportError as exc:
            raise RuntimeError(
                "Missing kaggle_environments. Install the pinned Ubuntu requirements before training."
            ) from exc
        self.env = make(
            "orbit_wars",
            configuration={"seed": seed, "randomSeed": seed, "episodeSteps": episode_steps},
            debug=True,
        )
        self.env.reset(num_agents=2)
        # 先走一步空动作，把 reset 后的初始态推进到真正可对局的活动态
        self.states = self.env.step([[], []])
        self.done = False

    def step(self, actions0: list[list[float]], actions1: list[list[float]]) -> tuple[dict, dict, bool]:
        self.states = self.env.step([actions0, actions1])
        self.done = all(state.get("status", "ACTIVE") != "ACTIVE" for state in self.states)
        return self.states[0].get("observation", {}), self.states[1].get("observation", {}), self.done

    @property
    def rewards(self) -> tuple[float, float]:
        return tuple(float(state.get("reward", 0.0) or 0.0) for state in self.states)  # type: ignore[return-value]


class Agent:
    """把网络输出转成官方动作。deterministic=True 走贪心 argmax（评估/推理），
    False 走按概率采样（训练时探索）。返回 (动作, 经验)；经验用于之后的 PPO 更新。"""

    def __init__(self, model: TargetPolicy, device: torch.device) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def act(self, observation: dict[str, Any], player_id: int, deterministic: bool) -> tuple[list[list[float]], dict | None]:
        state = extract_state(observation, player_id)
        if state is None or not state["candidate_mask"].any():
            return [], None   # 无可行动作：这步等待
        sf = torch.from_numpy(state["self_features"]).unsqueeze(0).to(self.device)
        cf = torch.from_numpy(state["candidate_features"]).unsqueeze(0).to(self.device)
        gf = torch.from_numpy(state["global_features"]).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(state["candidate_mask"]).unsqueeze(0).to(self.device)
        logits, values = self.model(sf.float(), cf.float(), gf.float(), mask.bool())
        distribution = Categorical(logits=logits.squeeze(0))
        action_index = logits.argmax(dim=1).item() if deterministic else distribution.sample().item()
        if not state["candidate_mask"][action_index]:
            return [], None
        source_id = int(state["planets"][state["source_index"], 0])
        action = [[source_id, float(state["angles"][action_index]), state["source_ships"]]]   # 全军出击
        # 记录这一步的输入/所选动作/对数概率/价值估计，供 PPO 复算新旧策略比
        experience = {
            "self_features": state["self_features"].copy(),
            "candidate_features": state["candidate_features"].copy(),
            "global_features": state["global_features"].copy(),
            "candidate_mask": state["candidate_mask"].copy(),
            "action": action_index,
            "log_prob": distribution.log_prob(torch.tensor(action_index, device=self.device)).item(),
            "value": values.item(),
        }
        return action, experience


class RolloutBuffer:
    """经验缓冲：按时间顺序攒下每步的 (经验, 奖励, 是否终局)，供一个迭代结束后统一更新。"""

    def __init__(self) -> None:
        self.experiences: list[dict] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def add(self, experience: dict | None, reward: float, done: bool) -> None:
        if experience is not None:   # 只记录真正发了兵的步（等待步没有可训练的经验）
            self.experiences.append(experience)
            self.rewards.append(float(reward))
            self.dones.append(done)

    def end_episode(self, final_reward: float) -> None:
        # 用环境给的真实胜负奖励覆盖最后一步的塑形奖励，并把它标为终局
        if self.rewards:
            self.rewards[-1] = float(final_reward)
            self.dones[-1] = True

    def extend(self, other: "RolloutBuffer") -> None:
        self.experiences.extend(other.experiences)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)

    def clear(self) -> None:
        self.experiences.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


def compute_gae(rewards: list[float], values: list[float], dones: list[bool], gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """广义优势估计(GAE)。从后往前递推：先算单步 TD 误差 delta，再按 γλ 指数衰减累加成
    优势 advantage；returns = advantage + value 作为价值头的回归目标。终局处 nonterminal=0
    会切断向未来的传播（终局之后没有下一状态）。"""
    returns = np.zeros(len(rewards), dtype=np.float32)
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0          # 累加器，存"上一步（更靠后）"算出的优势 A_{t+1}
    next_value = 0.0   # 下一状态的价值 V(s_{t+1})，从末步往前逐步填；末步之后无未来，取 0
    for index in reversed(range(len(rewards))):   # 必须倒序：A_t 依赖已算好的 A_{t+1}
        nonterminal = 1.0 - float(dones[index])   # 终局→0（切断未来），非终局→1
        # 单步 TD 误差 δ_t = r_t + γ·V(s_{t+1})·nonterminal − V(s_t)
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        # 递推优势 A_t = δ_t + γλ·nonterminal·A_{t+1}：把后续优势按 γλ 指数衰减累加进来
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        returns[index] = gae + values[index]      # 回归目标 return_t = A_t + V(s_t)，供价值头拟合
        next_value = values[index]                # 本步的 V(s_t) 即上一步（更靠前）的 V(s_{t+1})
    return returns, advantages


def ppo_update(model: TargetPolicy, optimizer: torch.optim.Optimizer, buffer: RolloutBuffer, device: torch.device, cfg: Config) -> dict[str, float]:
    """PPO 更新：算 GAE → 把变长的候选集补零对齐成规整张量 → 多轮小批量优化。
    损失 = 截断策略损失 + value_coef·价值损失 − entropy_coef·熵（熵项鼓励探索）。"""
    if not buffer:
        return {"policy": 0.0, "value": 0.0, "entropy": 0.0}
    # ── ① 用整批经验算 GAE：advantages 供策略损失加权，returns 作价值头回归目标 ──
    values = [item["value"] for item in buffer.experiences]
    returns, advantages = compute_gae(buffer.rewards, values, buffer.dones, cfg.gamma, cfg.gae_lambda)

    # ── ② 变长候选补零对齐：各步候选数不同，取最大候选数，把每步特征/掩码/动作补零(pad)到同一形状，才能堆成规整批张量 ──
    max_candidates = max(item["candidate_features"].shape[0] for item in buffer.experiences)
    count = len(buffer)
    sf = np.zeros((count, SELF_DIM), dtype=np.float32)                 # 来源特征
    cf = np.zeros((count, max_candidates, CAND_DIM), dtype=np.float32) # 候选特征（不足 max_candidates 处留 0）
    gf = np.zeros((count, GLOBAL_DIM), dtype=np.float32)               # 全局特征
    mask = np.zeros((count, max_candidates), dtype=bool)              # 有效候选掩码（补零处保持 False，前向时会被屏蔽）
    actions = np.zeros(count, dtype=np.int64)                         # 当时实际选中的候选索引
    old_log_probs = np.zeros(count, dtype=np.float32)                # 采样时旧策略的 log π（算概率比用）
    for index, item in enumerate(buffer.experiences):                # 逐步把真实长度的数据填进补零张量
        candidates = item["candidate_features"].shape[0]
        sf[index] = item["self_features"]
        cf[index, :candidates] = item["candidate_features"]
        gf[index] = item["global_features"]
        mask[index, :candidates] = item["candidate_mask"]
        actions[index] = item["action"]
        old_log_probs[index] = item["log_prob"]

    # ── ③ 一次性搬上目标设备，并标准化优势（均值0/方差1，稳定训练） ──
    tensors = [
        torch.from_numpy(array).to(device)
        for array in (sf, cf, gf, mask, actions, old_log_probs, returns, advantages)
    ]
    sf_t, cf_t, gf_t, mask_t, actions_t, old_log_probs_t, returns_t, advantages_t = tensors
    advantages_t = (advantages_t.float() - advantages_t.float().mean()) / (advantages_t.float().std(unbiased=False) + 1e-8)  # 优势标准化，稳定训练

    # ── ④ 多轮小批量 PPO 优化：同一批数据反复用 ppo_epochs 轮 ──
    order = np.arange(count)
    totals = np.zeros(3, dtype=np.float64)   # 累计 policy/value/entropy 三项损失，末尾取均值打印
    updates = 0
    model.train()
    for _ in range(cfg.ppo_epochs):        # 同一批数据重复优化 ppo_epochs 轮
        np.random.shuffle(order)           # 每轮打乱样本顺序再切小批
        for start in range(0, count, cfg.batch_size):
            indices = torch.as_tensor(order[start : start + cfg.batch_size], device=device)
            # 用当前（新）策略重新前向这批样本，拿到新 logits 与价值预测
            logits, predicted_values = model(sf_t[indices].float(), cf_t[indices].float(), gf_t[indices].float(), mask_t[indices].bool())
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions_t[indices].long())   # 新策略对"当时所选动作"的 log π
            entropy = distribution.entropy().mean()                            # 策略熵，越大越随机（用于鼓励探索）
            ratio = (new_log_probs - old_log_probs_t[indices].float()).exp()   # 新旧策略概率比 r = exp(logπ_new − logπ_old)
            # 策略损失（PPO 截断目标）：把 r 夹在 [1-ε, 1+ε] 内取较小者，限制单次更新步幅；前置负号把"最大化目标"变成"最小化损失"
            policy_loss = -torch.minimum(ratio * advantages_t[indices], ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages_t[indices]).mean()
            value_loss = F.mse_loss(predicted_values, returns_t[indices].float())   # 价值损失：价值头预测 vs GAE 回报的 MSE
            # 总损失 = 策略损失 + value_coef·价值损失 − entropy_coef·熵（熵项取负 = 奖励高熵、鼓励探索）
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)   # 清掉上一批的梯度
            loss.backward()                          # 反向传播求梯度
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)   # 梯度裁剪，防梯度爆炸
            optimizer.step()                         # 按梯度走一步，更新参数
            totals += (policy_loss.item(), value_loss.item(), entropy.item())
            updates += 1
    model.eval()
    return dict(zip(("policy", "value", "entropy"), totals / max(updates, 1)))   # 返回三项损失均值供日志


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    # 深拷一份权重到 CPU：给对手池存档、保存 best 状态用，避免与在训模型共享同一份张量。
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def shaped_reward(observation: dict[str, Any], player_id: int) -> float:
    """塑形奖励：对"下一状态"打一个绝对局面分 = 兵力差 + 星球数差。兵力 = 星球驻军 + 在途舰队，
    与官方计分口径一致（发出的大舰队仍计入己方兵力，不会被误判为丢兵）。终局那一步会在
    end_episode 里被环境的真实胜负奖励覆盖。"""
    planets = np.asarray(observation.get("planets", []), dtype=np.float32)
    if len(planets) == 0:
        return 0.0
    # fleets 每行 = [id, owner, x, y, angle, from_planet_id, ships]，owner 在第 1 列、ships 在第 6 列
    fleets = np.asarray(observation.get("fleets", []), dtype=np.float32)
    alive = planets[:, 0] >= 0
    mine = alive & (planets[:, 1] == player_id)
    enemy = alive & (planets[:, 1] >= 0) & (planets[:, 1] != player_id)
    my_ships = planets[mine, 5].sum()      # 己方星球驻军
    enemy_ships = planets[enemy, 5].sum()  # 敌方星球驻军
    if len(fleets):                        # 再加上各自在途的舰队兵力
        my_ships += fleets[fleets[:, 1] == player_id, 6].sum()
        enemy_ships += fleets[fleets[:, 1] != player_id, 6].sum()
    ship_gap = (my_ships - enemy_ships) / 2000.0     # 兵力差（除数只是缩放）
    planet_gap = (mine.sum() - enemy.sum()) / 100.0  # 星球数差
    return float(ship_gap + planet_gap)


def evaluate(model: TargetPolicy, reference_state: dict[str, torch.Tensor], device: torch.device, cfg: Config, seeds: list[int]) -> float:
    """让当前模型和参考（best）模型贪心对打若干局，返回当前模型胜率。轮换座位以抵消先后手偏差。"""
    reference = TargetPolicy(cfg.hidden_size, cfg.num_blocks, cfg.dropout).to(device)
    reference.load_state_dict(reference_state)
    reference.eval()
    current_agent, reference_agent = Agent(model, device), Agent(reference, device)
    wins = 0
    for game_index, seed in enumerate(seeds):
        game = GameRunner(seed, cfg.episode_steps)
        learner_player = game_index % 2   # 轮流当 0/1 号位
        while not game.done:
            obs0 = game.states[0].get("observation", {})
            obs1 = game.states[1].get("observation", {})
            if learner_player == 0:
                actions0, _ = current_agent.act(obs0, 0, deterministic=True)
                actions1, _ = reference_agent.act(obs1, 1, deterministic=True)
            else:
                actions0, _ = reference_agent.act(obs0, 0, deterministic=True)
                actions1, _ = current_agent.act(obs1, 1, deterministic=True)
            game.step(actions0, actions1)
        reward = game.rewards[learner_player]
        wins += int(reward > 0)
    return wins / len(seeds)


def save_checkpoint(path: Path, model: TargetPolicy, optimizer: torch.optim.Optimizer, iteration: int, metric: float, cfg: Config) -> None:
    # 存 .pt 检查点，用于存档与续训。format / architecture / action_contract 都是自描述元数据
    # （记录这份检查点出自哪个赛制、什么结构、什么动作契约），供人查阅，代码不读它们。
    torch.save(
        {
            "format": "orbit-wars-2p-training-checkpoint-v1",
            "iteration": iteration,
            "metric": metric,
            "model_state_dict": model.state_dict(),   # 权重本体；提交端用的 .npz 由同一份 state_dict 导出
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": {"hidden_size": cfg.hidden_size, "num_blocks": cfg.num_blocks, "dropout": cfg.dropout},
            "action_contract": "richest-owned-planet -> one target -> all ships",
        },
        path,
    )


def export_submission_weights(model: TargetPolicy, path: Path) -> Path:
    """把当前权重导出成提交端能直接读的 model_weights.npz。

    提交端不 import torch、只认 .npz，所以 .pt 不能直接当提交权重：这里把 state_dict 里的
    每个张量转成 NumPy 数组存进 npz，键名保持模块路径（如 self_encoder.input_proj.weight）
    原样不动——提交端正是按这些键名取参数的。
    """
    path.parent.mkdir(parents=True, exist_ok=True)   # 单独把训练脚本放到别处跑时，目标目录可能不存在
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    np.savez(path, **arrays)
    return path


def train(cfg: Config) -> None:
    """训练主循环：从随机权重起，每个迭代用历史对手池自我对弈若干局收集经验 → PPO 更新 →
    定期评估并存档（胜率达标才替换 best）、定期把当前权重存入对手池。"""
    # ── ① 准备：固定随机种子、选设备、建存档目录并把本次超参落盘 ──
    set_seed(cfg.seed)   # 固定 Python/NumPy/Torch 随机源，尽量可复现
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 有 GPU 用 GPU，否则退回 CPU
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()}, handle, indent=2)   # 存一份超参快照便于复盘

    # ── ② 建模型与优化器：从随机初始化起（无预训练权重） ──
    model = TargetPolicy(cfg.hidden_size, cfg.num_blocks, cfg.dropout).to(device)
    model.eval()   # 采样对局时用 eval（关 dropout）；只有 ppo_update 内部才临时切回 train
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ── ③ 自我对弈记账：对手池 + best 快照 + 经验缓冲 ──
    # 对手池：初始只有随机权重的自己；训练中不断把新版本加入，从中随机抽对手，避免只会打一种对手
    opponent_pool: deque[dict[str, torch.Tensor]] = deque([clone_state(model)], maxlen=cfg.opponent_pool_size)
    best_state = clone_state(model)   # 目前最强权重的快照，作为评估的参照对手
    best_win_rate = 0.0
    rng = np.random.default_rng(cfg.seed)   # 抽对手 / 生成开局种子用的独立随机源
    buffer = RolloutBuffer()   # 跨整个迭代累计经验，ppo_update 之后清空
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} output={cfg.output_dir}")

    for iteration in range(1, cfg.total_iters + 1):
        started = time.perf_counter()
        terminal_rewards: list[float] = []

        # ── ④ 自我对弈：本迭代打 games_per_iter 局，把每局经验汇进 buffer ──
        for game_index in range(cfg.games_per_iter):
            # 每局随机抽一个历史版本当对手（对手权重固定、不参与训练）
            opponent_model = TargetPolicy(cfg.hidden_size, cfg.num_blocks, cfg.dropout).to(device)
            opponent_model.load_state_dict(opponent_pool[int(rng.integers(len(opponent_pool)))])
            opponent_model.eval()
            learner, opponent = Agent(model, device), Agent(opponent_model, device)   # learner=在训模型，opponent=抽到的历史版本
            learner_player = game_index % 2   # 轮换座位（0/1 号位交替），抵消先后手偏差
            game = GameRunner(int(rng.integers(2**31 - 1)), cfg.episode_steps)   # 每局随机开局种子
            episode_buffer = RolloutBuffer()   # 单局经验，跑完再并进大 buffer
            while not game.done:               # 一局到底：每步双方各决策 → 推进环境 → 记 learner 的经验
                obs0 = game.states[0].get("observation", {})
                obs1 = game.states[1].get("observation", {})
                # learner 坐在 learner_player 号位；双方都用采样(deterministic=False)保留探索
                if learner_player == 0:
                    actions0, experience = learner.act(obs0, 0, deterministic=False)   # 只留 learner 那侧的 experience
                    actions1, _ = opponent.act(obs1, 1, deterministic=False)
                else:
                    actions0, _ = opponent.act(obs0, 0, deterministic=False)
                    actions1, experience = learner.act(obs1, 1, deterministic=False)
                next_obs0, next_obs1, done = game.step(actions0, actions1)   # 官方环境同时结算两边动作
                # 奖励用"这步动作之后的局面"来打分，对齐到刚才那步经验上
                next_observation = next_obs0 if learner_player == 0 else next_obs1
                episode_buffer.add(experience, shaped_reward(next_observation, learner_player), done)   # experience 为 None（等待步）时不记
            final_reward = game.rewards[learner_player]   # 官方给的这局真实胜负奖励
            episode_buffer.end_episode(final_reward)   # 末步的塑形奖励换成真实胜负奖励
            buffer.extend(episode_buffer)
            terminal_rewards.append(final_reward)

        # ── ⑤ PPO 更新：用本迭代经验更新一次，清空 buffer，并定期把当前权重存进对手池 ──
        stats = ppo_update(model, optimizer, buffer, device, cfg)   # 用这一迭代攒的经验更新
        samples = len(buffer)
        buffer.clear()   # 经验只用一轮（on-policy），用完即弃
        if iteration % cfg.opponent_update_freq == 0:   # 定期把当前权重快照进对手池，让对手也变强
            opponent_pool.append(clone_state(model))

        # ── ⑥ 定期评估与存档：贪心对打 best，胜率达标才刷新 best 并导出提交权重 ──
        if iteration % cfg.eval_interval == 0:
            eval_seeds = [cfg.seed + iteration * 10_000 + i for i in range(cfg.eval_games)]   # 固定种子，评估可比
            win_rate = evaluate(model, best_state, device, cfg, eval_seeds)   # 当前模型 vs best 的胜率
            save_checkpoint(cfg.output_dir / "latest.pt", model, optimizer, iteration, win_rate, cfg)   # 每次评估都存 latest（可续训）
            if win_rate >= 0.55:   # 明显强于当前 best 才替换（2P 要求胜率≥0.55）
                best_state = clone_state(model)
                best_win_rate = win_rate
                save_checkpoint(cfg.output_dir / "best.pt", model, optimizer, iteration, win_rate, cfg)
                # 只在刷新 best 时导出，submission_2p/ 里的权重因此始终等于当前最好的模型
                export_submission_weights(model, SUBMISSION_WEIGHTS_PATH)
            print(f"iter={iteration:5d} samples={samples:5d} terminal_mean={np.mean(terminal_rewards):+.3f} eval_vs_best={win_rate:.3f} best={best_win_rate:.3f}")
        elif iteration % 5 == 0:   # 非评估迭代每 5 次打印一次损失，便于盯训练是否健康
            print(f"iter={iteration:5d} samples={samples:5d} terminal_mean={np.mean(terminal_rewards):+.3f} policy={stats['policy']:.4f} value={stats['value']:.4f} entropy={stats['entropy']:.4f} sec={time.perf_counter() - started:.1f}")

    save_checkpoint(cfg.output_dir / "final.pt", model, optimizer, cfg.total_iters, best_win_rate, cfg)   # 训练结束存最终检查点


def parse_args() -> Config:
    # 命令行只暴露最常调的几个超参，其余用 Config 默认值。注意用命名参数构造 Config，避免字段错位。
    parser = argparse.ArgumentParser(description="Train a 2-player Orbit Wars PPO policy from random initialization.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/orbit_wars_2p"))
    parser.add_argument("--total-iters", type=int, default=2_000)
    parser.add_argument("--games-per-iter", type=int, default=8)
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
