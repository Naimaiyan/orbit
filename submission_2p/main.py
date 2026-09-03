"""2P（1v1）实战方案的纯 NumPy 推理入口。

把本文件和 model_weights.npz 一起打进提交包上传到 Kaggle。提交包刻意不 import
torch，也不 import kaggle_environments：Kaggle 每步喂进来观测 observation，只用
提交运行环境自带的 NumPy 手写一遍前向就够做确定性推理，既轻量又不依赖训练框架。
"""

import os
import math
from pathlib import Path

# 把 BLAS/OpenMP 线程数锁成 1：提交环境本身是单局串行推理，多线程只会增加调度开销、
# 还可能让浮点归约顺序不稳定；锁成单线程既省资源又让结果可复现。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np


BOARD_SIZE = 100.0            # 地图边长（正方形），特征里用它把坐标归一化到 0~1
BOARD_CENTER = (50.0, 50.0)   # 太阳/地图中心坐标
MAX_SHIPS = 500.0             # 单星球飞船数归一化上限
MAX_PROD = 20.0              # 生产值归一化上限
MAX_DISTANCE = math.sqrt(2.0) * BOARD_SIZE  # 地图上两点最大可能距离（对角线），用于距离归一化


def _agent_dir():
    """定位权重文件所在目录。正常情况取本文件所在目录；若 __file__ 不可用（某些
    打包/执行环境下会这样），回退到 Kaggle 提交固定的挂载路径。"""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path("/kaggle_simulations/agent")


# 模块加载时一次性读入权重：npz 里每个数组的键名 == 训练时 state_dict 的模块路径
# （如 "self_encoder.input_proj.weight"），下面手写前向就按这些键名去取参数。
_WEIGHTS_PATH = _agent_dir() / "model_weights.npz"
if not _WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Missing exported model weights: {_WEIGHTS_PATH}")
with np.load(_WEIGHTS_PATH, allow_pickle=False) as _archive:
    W = {name: _archive[name].astype(np.float32, copy=False) for name in _archive.files}


def _linear(x, prefix):
    # 手写全连接层。注意要对权重转置 .T：PyTorch 的 nn.Linear 把权重存成 [out, in]，
    # 而这里算的是 x @ weight.T + bias，与训练端逐位对齐。
    return x @ W[f"{prefix}.weight"].T + W[f"{prefix}.bias"]


def _layer_norm(x, prefix, eps=1e-5):
    # 手写 LayerNorm。eps=1e-5 必须与 PyTorch nn.LayerNorm 的默认值一致，否则数值会偏。
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * W[f"{prefix}.weight"] + W[f"{prefix}.bias"]


def _encoder(x, prefix):
    # 一个编码器 = 输入投影 input_proj + 若干残差块。残差块数量不写死：用
    # "while 该块的权重键还在 W 里" 自适应地跑完所有块，训练端加减层这里无需改动。
    x = _linear(x, f"{prefix}.input_proj")
    block_index = 0
    while f"{prefix}.blocks.{block_index}.norm.weight" in W:
        residual = x  # 残差直连：块的输出再加回输入
        x = _layer_norm(x, f"{prefix}.blocks.{block_index}.norm")
        x = np.maximum(_linear(x, f"{prefix}.blocks.{block_index}.linear1"), 0.0)  # ReLU
        x = _linear(x, f"{prefix}.blocks.{block_index}.linear2")
        x = residual + x
        block_index += 1
    return x


def _target_head(x):
    # 打分头：3 层全连接 + ReLU，最后输出每个候选目标的 1 维分数。索引取 0/3/6 是因为
    # 训练端用 nn.Sequential 把 Linear 夹在 ReLU/Dropout 之间，只有第 0/3/6 个子模块是 Linear。
    x = np.maximum(_linear(x, "target_head.0"), 0.0)
    x = np.maximum(_linear(x, "target_head.3"), 0.0)
    return _linear(x, "target_head.6").squeeze(-1)


def _fleet_speed(ships):
    # 舰队速度随兵数对数增长（兵越多飞得越快），再按 1.5 次幂加速，最后封顶到基础速度的 6 倍。
    # 仅用于算候选特征里的"粗略 ETA"，与官方移动规则的口径一致。
    ratio = min(math.log(max(float(ships), 1.0)) / math.log(1000.0), 1.0)
    return 1.0 + 5.0 * ratio**1.5


def _planet_features(planets, player_id):
    """把每颗星球编码成 14 维基础特征（全部归一化到大致 0~1，便于网络学习）。
    planets 每行是官方观测格式 [id, owner, x, y, radius, ships, production]。"""
    alive = planets[:, 0] >= 0            # id<0 表示该槽位是死星球（占位），用来过滤
    owner = planets[:, 1]                 # 所有者：本方 id / 对手 id / -1 中立
    x, y, radius, ships, production = (planets[:, i] for i in range(2, 7))
    dx, dy = x - BOARD_CENTER[0], y - BOARD_CENTER[1]   # 相对地图中心（太阳）的位移
    center_distance = np.sqrt(dx**2 + dy**2)
    orbit_radius = center_distance + radius
    # 近太阳的星球会绕轨道移动：轨道半径够小且不在正中心，判定为"绕行中"
    orbiting = (orbit_radius < 50.0) & (center_distance > 0.5)
    features = np.zeros((len(planets), 14), dtype=np.float32)
    features[:, 0] = x / BOARD_SIZE                          # 0-1: 归一化坐标
    features[:, 1] = y / BOARD_SIZE
    features[:, 2] = radius / 5.0                            # 2: 归一化半径
    features[:, 3] = np.clip(ships, 0.0, MAX_SHIPS) / MAX_SHIPS       # 3: 归一化驻军
    features[:, 4] = np.clip(production, 0.0, MAX_PROD) / MAX_PROD    # 4: 归一化生产值
    features[:, 5] = center_distance / 70.7                  # 5: 离中心的距离（70.7≈半对角线）
    features[:, 6] = orbiting.astype(np.float32)             # 6: 是否在绕轨道
    features[:, 7] = dx / BOARD_SIZE                         # 7-8: 相对中心的方向分量
    features[:, 8] = dy / BOARD_SIZE
    features[:, 9] = ((owner == player_id) & alive).astype(np.float32)                    # 9: 是我方
    features[:, 10] = ((owner >= 0) & (owner != player_id) & alive).astype(np.float32)    # 10: 是敌方
    features[:, 11] = ((owner == -1) & alive).astype(np.float32)                          # 11: 是中立
    features[:, 12] = np.where(orbiting, orbit_radius / BOARD_SIZE, 0.0)                  # 12: 绕行星球的轨道半径
    features[:, 13] = alive.astype(np.float32)                                            # 13: 存活标志
    return features


def _choose_action(observation):
    """本方案的动作契约（2P）：只从"己方飞船最多的一颗"星球出发 → 在其余存活星球里
    选 1 个目标 → 全军出击。返回 [[出发星球ID, 方向弧度, 飞船数]]，无可行动作则返回 []（等待）。

    这就是训练端 extract_state（拼三路特征）+ Agent.act（前向 + 贪心选目标）合到一起的推理版，
    只是把 torch 前向换成本文件手写的 NumPy 前向；特征与网络计算必须与训练端逐位一致。"""
    # ── ① 选来源与候选：落实动作契约的"选来源、列候选"（对应训练端 extract_state 前半段） ──
    planets_list = observation.get("planets", [])
    if not planets_list:
        return []
    player_id = int(observation.get("player", 0))
    planets = np.asarray(planets_list, dtype=np.float32)
    alive = planets[:, 0] >= 0
    owner, ships = planets[:, 1], planets[:, 5]
    # 可作为出发点的星球：存活、我方、且驻军≥4（太少不值得发兵）
    mine = alive & (owner == player_id) & (ships >= 4.0)
    if not np.any(mine):
        return []
    # 唯一来源 = 我方驻军最多的那颗星球
    source_candidates = np.flatnonzero(mine)
    source_index = source_candidates[np.argmax(ships[source_candidates])]
    # 候选目标 = 除来源外的所有存活星球
    target_indices = np.flatnonzero(alive)
    target_indices = target_indices[target_indices != source_index]
    if len(target_indices) == 0:
        return []

    # ── ② 拼候选特征：18 维 = 14 维目标基础特征 + 4 维"从来源瞄准它"的几何量 ──
    features = _planet_features(planets, player_id)   # 全场星球的归一化行表；下面网络只吃这张表，不再碰原始 planets
    source_x, source_y = planets[source_index, 2], planets[source_index, 3]   # 原始坐标仅在此用于算相对几何
    dx = planets[target_indices, 2] - source_x        # 来源→各目标的原始位移（还没归一化）
    dy = planets[target_indices, 3] - source_y
    distance = np.sqrt(dx**2 + dy**2)
    candidate_features = np.zeros((len(target_indices), 18), dtype=np.float32)
    candidate_features[:, :14] = features[target_indices]      # 前 14 列：直接搬候选目标那几行的归一化特征
    candidate_features[:, 14] = dx / BOARD_SIZE                 # 14-15: 指向目标的方向分量（÷边长压到约 -1~1）
    candidate_features[:, 15] = dy / BOARD_SIZE
    candidate_features[:, 16] = distance / MAX_DISTANCE         # 16: 到目标的距离（÷对角线压到 0~1）
    candidate_features[:, 17] = np.clip(distance / _fleet_speed(ships[source_index]), 0.0, 50.0) / 50.0  # 17: 粗略 ETA=距离/速度，再压到 0~1
    mask = distance > 1e-6   # 距离≈0 的目标（与来源重合）排除，避免除零和自打自
    if not np.any(mask):
        return []

    # ── ③ 拼全局特征：8 维 = 步数进度 + 我/敌/中立星球数 + 我/敌总驻军 + 我/敌总生产 ──
    enemy = alive & (owner >= 0) & (owner != player_id)
    neutral = alive & (owner == -1)
    global_features = np.asarray(
        [
            observation.get("step", 0) / 500.0,                       # 0: 步数进度（当前步/最大步）
            mine.sum() / 64.0,                                       # 1: 我方星球数
            enemy.sum() / 64.0,                                      # 2: 敌方星球数
            neutral.sum() / 64.0,                                    # 3: 中立星球数
            np.clip(ships[mine].sum(), 0.0, 32000.0) / 32000.0,      # 4: 我方总驻军
            np.clip(ships[enemy].sum(), 0.0, 32000.0) / 32000.0,     # 5: 敌方总驻军
            np.clip(planets[mine, 6].sum(), 0.0, 1280.0) / 1280.0,   # 6: 我方总生产值
            np.clip(planets[enemy, 6].sum(), 0.0, 1280.0) / 1280.0,  # 7: 敌方总生产值
        ],
        dtype=np.float32,
    )

    # ── ④ 三路前向：分别编码来源/候选/全局，每个候选拼上"来源+全局"再送打分头（对应训练端网络 forward） ──
    self_h = _encoder(features[source_index][None, :], "self_encoder")[0]   # 来源隐向量
    candidate_h = _encoder(candidate_features, "candidate_encoder")          # 各候选隐向量
    global_h = _encoder(global_features[None, :], "global_encoder")[0]      # 全局隐向量
    joint = np.concatenate(   # 把"来源+全局"广播到每个候选后与候选隐向量拼接
        [np.repeat(self_h[None, :], len(target_indices), axis=0), np.repeat(global_h[None, :], len(target_indices), axis=0), candidate_h],
        axis=-1,
    )

    # ── ⑤ 选目标并还原成官方动作：屏蔽非法目标 → 贪心 argmax → 拼 [出发星球ID, 方向, 兵数] ──
    # logit_scale 是训练时采样用的全局温度；这里贪心 argmax 取最大分，正数缩放不改变谁最大，
    # 所以它对最终选择没有影响，保留只是为了与训练端逐位一致。
    logits = _target_head(joint) * float(np.clip(W["logit_scale"], 0.1, 10.0))
    logits[~mask] = -np.inf                       # 屏蔽非法目标（与来源重合的候选）
    target_slot = int(np.argmax(logits))          # 选分数最高的目标（推理无采样，纯贪心）
    source_id = int(planets[source_index, 0])
    angle = float(math.atan2(float(dy[target_slot]), float(dx[target_slot])))  # 出发点指向该目标的方向（弧度）
    return [[source_id, angle, int(ships[source_index])]]                      # 全军出击：来源全部驻军


def agent(observation, configuration):
    """Kaggle 每步调用的入口。configuration 用不到，故留空不用。"""
    try:
        return _choose_action(observation)
    except Exception:
        # 出现任何意外（观测格式异常等）时返回合法的"空动作"而不是抛错——
        # 抛错会直接判负，返回空动作只是这一步不发兵，更稳。
        return []
