"""4P（四人自由混战 FFA）实战方案的纯 NumPy 推理入口。

与 2P 提交端同一套手写前向思路，差别集中在动作契约：4P 可从多颗来源星球里挑、
可以显式选择"等待不发兵"（hold）、全局特征也更宽（要描述三个对手）。详见 _choose_action。
把本文件和 model_weights.npz 一起打进提交包上传 Kaggle；同样不 import torch。
"""

import os
import math
from pathlib import Path

# 锁单线程：单局串行推理下多线程只增开销、还可能扰动浮点归约顺序，锁成 1 更稳更可复现。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np


BOARD_SIZE = 100.0            # 地图边长
BOARD_CENTER = (50.0, 50.0)   # 太阳/地图中心
MAX_SHIPS = 500.0             # 驻军归一化上限
MAX_PROD = 20.0              # 生产值归一化上限
MAX_SOURCES = 4              # 4P 每步最多考虑 4 颗来源星球（取驻军最多的 4 颗）
MAX_DISTANCE = math.sqrt(2.0) * BOARD_SIZE  # 对角线，距离归一化用


def _agent_dir():
    """定位权重目录：优先本文件目录，__file__ 不可用时回退 Kaggle 挂载路径。"""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path("/kaggle_simulations/agent")


# 模块加载时读入权重；键名 == 训练 state_dict 的模块路径，前向按键名取参数。
_WEIGHTS_PATH = _agent_dir() / "model_weights.npz"
if not _WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Missing exported model weights: {_WEIGHTS_PATH}")
with np.load(_WEIGHTS_PATH, allow_pickle=False) as _archive:
    W = {name: _archive[name].astype(np.float32, copy=False) for name in _archive.files}


def _linear(x, prefix):
    # 手写全连接：nn.Linear 权重存成 [out, in]，故这里要 .T 才是 x @ W.T + b。
    return x @ W[f"{prefix}.weight"].T + W[f"{prefix}.bias"]


def _layer_norm(x, prefix, eps=1e-5):
    # 手写 LayerNorm；eps 与 PyTorch 默认 1e-5 保持一致。
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * W[f"{prefix}.weight"] + W[f"{prefix}.bias"]


def _encoder(x, prefix):
    # 输入投影 + 若干残差块；块数用 while 键名探测自适应，不写死。
    x = _linear(x, f"{prefix}.input_proj")
    block_index = 0
    while f"{prefix}.blocks.{block_index}.norm.weight" in W:
        residual = x
        x = _layer_norm(x, f"{prefix}.blocks.{block_index}.norm")
        x = np.maximum(_linear(x, f"{prefix}.blocks.{block_index}.linear1"), 0.0)
        x = _linear(x, f"{prefix}.blocks.{block_index}.linear2")
        x = residual + x
        block_index += 1
    return x


def _mlp(x, prefixes):
    # 通用多层 MLP：除最后一层外每层都接 ReLU。pair_head / hold_head 都复用它，
    # 只是传入的层名前缀（如 ["pair_head.0","pair_head.3","pair_head.6"]）不同。
    for prefix in prefixes[:-1]:
        x = np.maximum(_linear(x, prefix), 0.0)
    return _linear(x, prefixes[-1])


def _fleet_speed(ships):
    # 兵越多飞得越快（对数增长再 1.5 次幂、封顶 6 倍），仅用于候选特征的粗略 ETA。
    ratio = min(math.log(max(float(ships), 1.0)) / math.log(1000.0), 1.0)
    return 1.0 + 5.0 * ratio**1.5


def _body_features(planets, player_id):
    """把每颗星球编码成 14 维基础特征（含义与 2P 完全一致）。
    planets 每行 = [id, owner, x, y, radius, ships, production]，id<0 为死槽位。"""
    alive = planets[:, 0] >= 0
    owner = planets[:, 1]
    x, y, radius, ships, production = (planets[:, i] for i in range(2, 7))
    dx, dy = x - BOARD_CENTER[0], y - BOARD_CENTER[1]
    center_distance = np.sqrt(dx**2 + dy**2)
    orbit_radius = center_distance + radius
    orbiting = (orbit_radius < 50.0) & (center_distance > 0.5)   # 近太阳绕行判定
    features = np.zeros((len(planets), 14), dtype=np.float32)
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


def _choose_action(observation):
    """本方案的动作契约（4P，与 2P 的三点不同）：
    ① 来源不止一颗——取己方驻军最多的最多 4 颗星球；
    ② 在"来源×目标"的所有组合里评分，另有一个显式的 hold（等待不发兵）选项；
    ③ 选中组合则全军出击，选中 hold 则返回 []。全局特征也更宽（要描述三个对手）。

    这就是训练端 extract_state（拼三路特征）+ Agent.act（前向 + 贪心选组合/hold）合到一起的推理版，
    只是把 torch 前向换成本文件手写的 NumPy 前向；特征与网络计算必须与训练端逐位一致。"""
    # ── ① 选多来源与候选：落实动作契约的"选来源、列候选"（对应训练端 extract_state 前半段） ──
    planets_list = observation.get("planets", [])
    if not planets_list:
        return []
    player_id = int(observation.get("player", 0))
    planets = np.asarray(planets_list, dtype=np.float32)
    alive = planets[:, 0] >= 0
    ships = planets[:, 5]
    owned = alive & (planets[:, 1] == player_id) & (ships >= 4.0)
    source_indices = np.flatnonzero(owned)
    if len(source_indices) == 0:
        return []
    # 差别①：按驻军排序取最多的 MAX_SOURCES(=4) 颗作为来源，[::-1] 让驻军多的排在前
    source_indices = source_indices[np.argsort(ships[source_indices])[-MAX_SOURCES:]][::-1]
    target_indices = np.flatnonzero(alive)
    if len(target_indices) == 0:
        return []

    # ── ② 拼候选网格特征：[来源数, 目标数, 18]，每个"来源→目标"对 = 14 维目标特征 + 4 维几何量 ──
    features = _body_features(planets, player_id)   # 全场星球的归一化行表；下面网络只吃这张表，不再碰原始 planets
    source_features = features[source_indices]      # 取出来源那几行的归一化特征
    source_xy = planets[source_indices, 2:4]        # 原始坐标仅在此用于算相对几何
    target_xy = planets[target_indices, 2:4]
    dx = target_xy[None, :, 0] - source_xy[:, None, 0]   # 广播成 [来源, 目标] 的原始位移矩阵（还没归一化）
    dy = target_xy[None, :, 1] - source_xy[:, None, 1]
    distance = np.sqrt(dx**2 + dy**2)
    candidates = np.zeros((len(source_indices), len(target_indices), 18), dtype=np.float32)
    candidates[:, :, :14] = features[target_indices][None, :, :]     # 前 14 列：把候选目标的归一化行广播到每个来源
    candidates[:, :, 14] = dx / BOARD_SIZE                           # 14-15: 该来源指向该目标的方向（÷边长压到约 -1~1）
    candidates[:, :, 15] = dy / BOARD_SIZE
    candidates[:, :, 16] = distance / MAX_DISTANCE                   # 16: 到目标的距离（÷对角线压到 0~1）
    speeds = np.asarray([_fleet_speed(ships[index]) for index in source_indices], dtype=np.float32)
    candidates[:, :, 17] = np.clip(distance / speeds[:, None], 0.0, 50.0) / 50.0   # 17: 粗略 ETA=距离/速度，再压到 0~1
    # 合法组合：距离>0 且目标≠来源自身
    pair_mask = (distance > 1e-6) & (target_indices[None, :] != source_indices[:, None])
    if not np.any(pair_mask):
        return []

    # ── ③ 拼全局特征：16 维（差别③，要描述三个对手） ──
    # 先放步数进度；再对"自己+三个对手"各放 3 项（星球数/总驻军/总生产）；
    # 最后 3 项是三个对手各自"是否还有星球"的存活标志。for pid 第一个是自己，其余是对手。
    global_values = [observation.get("step", 0) / 500.0]   # 维度 0: 步数进度（当前步/最大步）
    for pid in [player_id] + [rival for rival in range(4) if rival != player_id]:
        owned_by_pid = alive & (planets[:, 1] == pid)
        global_values.extend([owned_by_pid.sum() / 64.0, np.clip(planets[owned_by_pid, 5].sum(), 0.0, 32000.0) / 32000.0, np.clip(planets[owned_by_pid, 6].sum(), 0.0, 1280.0) / 1280.0])   # 该玩家的[星球数, 总驻军, 总生产值]
    for rival in [pid for pid in range(4) if pid != player_id]:
        global_values.append(float(np.any(alive & (planets[:, 1] == rival))))   # 该对手存活标志（1=还有星球）
    global_features = np.asarray(global_values, dtype=np.float32)

    # ── ④ 三路前向：编码来源/候选/全局，每个"来源×目标"对拼上"该来源+全局+该候选"隐向量（对应训练端网络 forward） ──
    source_h = _encoder(source_features, "source_encoder")   # 各来源隐向量
    candidate_h = _encoder(candidates, "target_encoder")     # 候选网格隐向量
    global_h = _encoder(global_features[None, :], "global_encoder")[0]   # 全局隐向量
    source_count, target_count = pair_mask.shape
    joint = np.concatenate(
        [
            np.repeat(source_h[:, None, :], target_count, axis=1),
            np.broadcast_to(global_h, (source_count, target_count, len(global_h))),
            candidate_h,
        ],
        axis=-1,
    )

    # ── ⑤ pair/hold 打分并还原动作：所有组合分数拉平、末尾接 hold 分数 → 贪心 argmax → 还原动作 ──
    # pair 分数和 hold 分数共用同一个温度 scale（贪心 argmax 下正数缩放不改变谁最大）
    scale = float(np.clip(W["logit_scale"], 0.1, 10.0))
    pair_logits = _mlp(joint, ["pair_head.0", "pair_head.3", "pair_head.6"]).squeeze(-1) * scale   # 每个(来源,目标)对的分数
    pair_logits[~pair_mask] = -np.inf                          # 屏蔽非法组合（目标=来源自身）
    # 差别②：hold（等待）分支。用"各来源隐向量的均值 + 全局隐向量"过 hold_head 得 1 个分数
    hold_input = np.concatenate([source_h.mean(axis=0), global_h])[None, :]
    hold_logit = float(_mlp(hold_input, ["hold_head.0", "hold_head.2"]).squeeze()) * scale
    # 把所有 pair 分数拉平、末尾接上 hold 分数，一起做 argmax（推理无采样，纯贪心）
    flat_logits = np.concatenate([pair_logits.reshape(-1), np.asarray([hold_logit], dtype=np.float32)])
    action_index = int(np.argmax(flat_logits))
    if action_index == source_count * target_count:           # 命中最后一个（hold）→ 这步等待
        return []
    source_slot, target_slot = divmod(action_index, target_count)   # 还原成 (来源槽, 目标槽)
    source_index = source_indices[source_slot]
    return [[int(planets[source_index, 0]), float(math.atan2(float(dy[source_slot, target_slot]), float(dx[source_slot, target_slot]))), int(ships[source_index])]]   # 全军出击：来源全部驻军


def agent(observation, configuration):
    """Kaggle 每步调用的入口。configuration 用不到，故留空不用。"""
    try:
        return _choose_action(observation)
    except Exception:
        # 意外时返回合法空动作而非抛错——抛错会判负，空动作只是这步不发兵，更稳。
        return []
