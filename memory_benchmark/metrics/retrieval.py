"""
检索指标模块
提供 R@K, NDCG@K, MRR 等检索评估指标
"""

import math
from typing import Sequence


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """计算 DCG@K (Discounted Cumulative Gain)

    DCG = sum(g_i / log2(i + 1)) for i in 1 to k

    Args:
        gains: 相关性得分序列 (1=相关, 0=不相关)
        k: 截断位置

    Returns:
        DCG@K 值
    """
    dcg = 0.0
    for i, g in enumerate(gains[:k]):
        dcg += g / math.log2(i + 2)  # i+2 because i is 0-indexed
    return dcg


def ndcg_at_k(
    retrieved: list[str],
    ground_truth: set[str],
    k: int
) -> float:
    """计算 NDCG@K (Normalized Discounted Cumulative Gain)

    NDCG = DCG / IDCG

    Args:
        retrieved: 检索结果 ID 列表（按相关性排序）
        ground_truth: 正确答案 ID 集合
        k: 截断位置

    Returns:
        NDCG@K 值 (0-1)
    """
    if len(ground_truth) == 0:
        return 0.0

    # 构建 gains 序列（检索结果中对应的相关性得分）
    gains = [1.0 if doc_id in ground_truth else 0.0 for doc_id in retrieved[:k]]
    dcg = dcg_at_k(gains, k)

    # 计算 IDCG（理想情况下的 DCG）
    ideal_gains = [1.0] * min(len(ground_truth), k)
    idcg = dcg_at_k(ideal_gains, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def recall_at_k(
    retrieved: list[str],
    ground_truth: set[str],
    k: int
) -> float:
    """计算 Recall@K

    Recall = |Retrieved ∩ GroundTruth| / |GroundTruth|

    Args:
        retrieved: 检索结果 ID 列表
        ground_truth: 正确答案 ID 集合
        k: 截断位置

    Returns:
        Recall@K 值 (0-1)
    """
    if len(ground_truth) == 0:
        return 0.0

    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & ground_truth)
    return hits / len(ground_truth)


def recall_any_at_k(
    retrieved: list[str],
    ground_truth: set[str],
    k: int
) -> float:
    """计算任意命中 Recall@K

    只要 top-k 中有一个命中即返回 1.0

    Args:
        retrieved: 检索结果 ID 列表
        ground_truth: 正确答案 ID 集合
        k: 截断位置

    Returns:
        1.0 如果有命中，否则 0.0
    """
    return 1.0 if len(set(retrieved[:k]) & ground_truth) > 0 else 0.0


def precision_at_k(
    retrieved: list[str],
    ground_truth: set[str],
    k: int
) -> float:
    """计算 Precision@K

    Precision = |Retrieved ∩ GroundTruth| / k

    Args:
        retrieved: 检索结果 ID 列表
        ground_truth: 正确答案 ID 集合
        k: 截断位置

    Returns:
        Precision@K 值 (0-1)
    """
    if k == 0:
        return 0.0

    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & ground_truth)
    return hits / k


def mrr(retrieved_list: list[list[str]], ground_truth_list: list[set[str]]) -> float:
    """计算 MRR (Mean Reciprocal Rank)

    MRR = (1/N) * sum(1/rank_i) for i in 1 to N
    其中 rank_i 是第一个正确答案的排名

    Args:
        retrieved_list: 每个查询的检索结果列表
        ground_truth_list: 每个查询的正确答案集合列表

    Returns:
        MRR 值 (0-1)
    """
    n = len(retrieved_list)
    if n == 0:
        return 0.0

    reciprocal_ranks = []
    for retrieved, ground_truth in zip(retrieved_list, ground_truth_list):
        rank = first_hit_rank(retrieved, ground_truth)
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / n


def first_hit_rank(retrieved: list[str], ground_truth: set[str]) -> int:
    """找到第一个正确答案的排名

    Args:
        retrieved: 检索结果 ID 列表
        ground_truth: 正确答案 ID 集合

    Returns:
        第一个命中的位置（1-indexed），未命中返回 0
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in ground_truth:
            return i + 1
    return 0


def average_precision(
    retrieved: list[str],
    ground_truth: set[str]
) -> float:
    """计算 AP (Average Precision)

    AP = (1/|GT|) * sum(P@i * rel_i) for i in 1 to |retrieved|

    Args:
        retrieved: 检索结果 ID 列表
        ground_truth: 正确答案 ID 集合

    Returns:
        AP 值 (0-1)
    """
    if len(ground_truth) == 0:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(retrieved):
        if doc_id in ground_truth:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / len(ground_truth)


def map(retrieved_list: list[list[str]], ground_truth_list: list[set[str]]) -> float:
    """计算 MAP (Mean Average Precision)

    MAP = (1/N) * sum(AP_i) for i in 1 to N

    Args:
        retrieved_list: 每个查询的检索结果列表
        ground_truth_list: 每个查询的正确答案集合列表

    Returns:
        MAP 值 (0-1)
    """
    n = len(retrieved_list)
    if n == 0:
        return 0.0

    aps = [average_precision(ret, gt) for ret, gt in zip(retrieved_list, ground_truth_list)]
    return sum(aps) / n


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    ground_truth_ids: list[str],
    k_values: list[int] = None
) -> dict:
    """计算完整的检索指标

    Args:
        retrieved_ids: 检索结果 ID 列表
        ground_truth_ids: 正确答案 ID 列表
        k_values: 要计算的 K 值列表

    Returns:
        包含各项指标的字典
    """
    if k_values is None:
        k_values = [1, 5, 10]

    ground_truth_set = set(ground_truth_ids)
    metrics = {}

    # Recall@K
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, ground_truth_set, k)
        metrics[f"recall_any@{k}"] = recall_any_at_k(retrieved_ids, ground_truth_set, k)

    # NDCG@K
    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, ground_truth_set, k)

    # MRR
    metrics["mrr"] = mrr([retrieved_ids], [ground_truth_set])

    # MAP
    metrics["map"] = map([retrieved_ids], [ground_truth_set])

    return metrics