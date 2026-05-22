"""
问答指标模块
提供 F1, EM, BERT-Score 等问答评估指标
"""

import re
import string
from typing import Callable
from collections import Counter


def normalize_text(text: str) -> str:
    """文本标准化

    - 转小写
    - 去除标点符号
    - 去除多余空格

    Args:
        text: 原始文本

    Returns:
        标准化后的文本
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """分词

    Args:
        text: 输入文本

    Returns:
        词列表
    """
    return normalize_text(text).split()


def f1_score(prediction: str, ground_truth: str) -> float:
    """计算 F1 分数（基于词级别）

    F1 = 2 * Precision * Recall / (Precision + Recall)

    Args:
        prediction: 预测答案
        ground_truth: 标准答案

    Returns:
        F1 分数 (0-1)
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # 计算交集
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def exact_match(prediction: str, ground_truth: str) -> float:
    """计算精确匹配分数

    Args:
        prediction: 预测答案
        ground_truth: 标准答案

    Returns:
        1.0 如果完全匹配，否则 0.0
    """
    pred_normalized = normalize_text(prediction)
    gt_normalized = normalize_text(ground_truth)
    return 1.0 if pred_normalized == gt_normalized else 0.0


def partial_match(prediction: str, ground_truth: str) -> float:
    """计算部分匹配分数

    检查预测答案是否包含在标准答案中，或反之

    Args:
        prediction: 预测答案
        ground_truth: 标准答案

    Returns:
        0.0 到 1.0 之间的分数
    """
    pred_normalized = normalize_text(prediction)
    gt_normalized = normalize_text(ground_truth)

    if not pred_normalized or not gt_normalized:
        return 0.0

    # 互包含检查
    if pred_normalized in gt_normalized:
        return len(pred_normalized) / len(gt_normalized)
    elif gt_normalized in pred_normalized:
        return len(gt_normalized) / len(pred_normalized)

    # 使用子串匹配率
    max_overlap = 0.0
    for i in range(len(pred_normalized) - 3):
        for j in range(i + 4, len(pred_normalized) + 1):
            substr = pred_normalized[i:j]
            if substr in gt_normalized:
                overlap = len(substr) / len(pred_normalized)
                max_overlap = max(max_overlap, overlap)

    return max_overlap


def rouge_l(prediction: str, ground_truth: str) -> float:
    """计算 ROUGE-L 分数

    ROUGE-L = LCS / max(len(pred), len(gt))

    Args:
        prediction: 预测答案
        ground_truth: 标准答案

    Returns:
        ROUGE-L 分数 (0-1)
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # 计算 LCS 长度
    m, n = len(pred_tokens), len(gt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    return lcs_length / max(m, n)


def bleu_score(prediction: str, ground_truth: str, n: int = 4) -> float:
    """计算 BLEU 分数（简化版）

    Args:
        prediction: 预测答案
        ground_truth: 标准答案
        n: 最大 n-gram 长度

    Returns:
        BLEU 分数 (0-1)
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # 计算 precision
    precisions = []
    for k in range(1, min(n + 1, len(pred_tokens) + 1)):
        pred_ngrams = Counter(tuple(pred_tokens[i:i+k]) for i in range(len(pred_tokens) - k + 1))
        gt_ngrams = Counter(tuple(gt_tokens[i:i+k]) for i in range(len(gt_tokens) - k + 1))

        matches = sum((pred_ngrams & gt_ngrams).values())
        total = len(pred_tokens) - k + 1

        if total > 0:
            precisions.append(matches / total)
        else:
            precisions.append(0.0)

    if not precisions:
        return 0.0

    # 计算几何平均
    geo_mean = 1.0
    for p in precisions:
        geo_mean *= p
    geo_mean = geo_mean ** (1.0 / len(precisions))

    # 简短惩罚
    bp = 1.0
    if len(pred_tokens) < len(gt_tokens):
        bp = math.exp(1 - len(gt_tokens) / max(len(pred_tokens), 1))

    return bp * geo_mean


import math


def compute_qa_metrics(
    prediction: str,
    ground_truth: str
) -> dict:
    """计算完整的问答指标

    Args:
        prediction: 预测答案
        ground_truth: 标准答案

    Returns:
        包含各项指标的字典
    """
    return {
        "f1": f1_score(prediction, ground_truth),
        "exact_match": exact_match(prediction, ground_truth),
        "partial_match": partial_match(prediction, ground_truth),
        "rouge_l": rouge_l(prediction, ground_truth),
        "bleu": bleu_score(prediction, ground_truth),
    }


def compute_batch_qa_metrics(
    predictions: list[str],
    ground_truths: list[str]
) -> dict:
    """计算批量问答指标的平均值

    Args:
        predictions: 预测答案列表
        ground_truths: 标准答案列表

    Returns:
        各项指标的平均值
    """
    n = len(predictions)
    if n == 0:
        return {
            "f1": 0.0,
            "exact_match": 0.0,
            "partial_match": 0.0,
            "rouge_l": 0.0,
            "bleu": 0.0,
        }

    total_f1 = sum(f1_score(p, g) for p, g in zip(predictions, ground_truths))
    total_em = sum(exact_match(p, g) for p, g in zip(predictions, ground_truths))
    total_pm = sum(partial_match(p, g) for p, g in zip(predictions, ground_truths))
    total_rl = sum(rouge_l(p, g) for p, g in zip(predictions, ground_truths))
    total_bleu = sum(bleu_score(p, g) for p, g in zip(predictions, ground_truths))

    return {
        "f1": total_f1 / n,
        "exact_match": total_em / n,
        "partial_match": total_pm / n,
        "rouge_l": total_rl / n,
        "bleu": total_bleu / n,
    }