"""
评估指标模块 - 异常检测评估
支持有标签和无标签两种评估模式
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)


def evaluate_detector(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    scores: np.ndarray
) -> Dict[str, Any]:
    """
    评估异常检测器性能

    Args:
        y_true: 真实标签（0=正常, 1=异常），无标签时为None
        y_pred: 预测标签
        scores: 异常分数

    Returns:
        评估指标字典
    """
    results = {
        'n_samples': len(y_pred),
        'n_anomalies_pred': int(np.sum(y_pred)),
        'anomaly_ratio_pred': float(np.sum(y_pred) / max(len(y_pred), 1))
    }

    if y_true is not None:
        # 有标签评估
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        scores = np.asarray(scores).flatten()

        # 基本指标
        results['n_anomalies_true'] = int(np.sum(y_true))
        results['anomaly_ratio_true'] = float(np.sum(y_true) / max(len(y_true), 1))

        # Precision / Recall / F1
        results['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        results['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        results['f1'] = float(f1_score(y_true, y_pred, zero_division=0))

        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results['true_positives'] = int(tp)
        results['false_positives'] = int(fp)
        results['true_negatives'] = int(tn)
        results['false_negatives'] = int(fn)

        # 检测率 (Detection Rate) = TP / (TP + FN) = Recall
        results['detection_rate'] = results['recall']

        # 误报率 (False Alarm Rate) = FP / (FP + TN)
        results['false_alarm_rate'] = float(fp / max(fp + tn, 1))

        # AUC-ROC
        try:
            results['auc_roc'] = float(roc_auc_score(y_true, scores))
        except ValueError:
            results['auc_roc'] = None

        # AUC-PR
        try:
            results['auc_pr'] = float(average_precision_score(y_true, scores))
        except ValueError:
            results['auc_pr'] = None

        # PR曲线数据
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
        results['pr_curve'] = {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist()
        }

    else:
        # 无标签评估 - 只返回分数统计
        results['score_mean'] = float(np.mean(scores))
        results['score_std'] = float(np.std(scores))
        results['score_min'] = float(np.min(scores))
        results['score_max'] = float(np.max(scores))
        results['score_median'] = float(np.median(scores))
        results['score_p95'] = float(np.percentile(scores, 95))
        results['score_p99'] = float(np.percentile(scores, 99))

    return results


def format_metrics_table(results: Dict[str, Any]) -> str:
    """格式化评估结果为表格字符串"""
    lines = []

    if 'precision' in results:
        # 有监督模式
        lines.append("=" * 50)
        lines.append("           异常检测评估结果 (有标签)")
        lines.append("=" * 50)
        lines.append(f"  样本总数:        {results['n_samples']}")
        lines.append(f"  真实异常数:      {results['n_anomalies_true']} ({results['anomaly_ratio_true']:.2%})")
        lines.append(f"  预测异常数:      {results['n_anomalies_pred']} ({results['anomaly_ratio_pred']:.2%})")
        lines.append("-" * 50)
        lines.append(f"  Precision:       {results['precision']:.4f}")
        lines.append(f"  Recall:         {results['recall']:.4f}")
        lines.append(f"  F1 Score:       {results['f1']:.4f}")
        lines.append("-" * 50)
        lines.append(f"  检测率:          {results['detection_rate']:.4f}")
        lines.append(f"  误报率:          {results['false_alarm_rate']:.4f}")
        if results.get('auc_roc') is not None:
            lines.append(f"  AUC-ROC:        {results['auc_roc']:.4f}")
        if results.get('auc_pr') is not None:
            lines.append(f"  AUC-PR:         {results['auc_pr']:.4f}")
        lines.append("-" * 50)
        lines.append(f"  TP: {results['true_positives']}  |  FP: {results['false_positives']}")
        lines.append(f"  TN: {results['true_negatives']}  |  FN: {results['false_negatives']}")
        lines.append("=" * 50)
    else:
        # 无监督模式
        lines.append("=" * 50)
        lines.append("           异常检测评估结果 (无标签)")
        lines.append("=" * 50)
        lines.append(f"  样本总数:        {results['n_samples']}")
        lines.append(f"  预测异常数:      {results['n_anomalies_pred']} ({results['anomaly_ratio_pred']:.2%})")
        lines.append("-" * 50)
        lines.append(f"  分数均值:        {results['score_mean']:.4f}")
        lines.append(f"  分数标准差:      {results['score_std']:.4f}")
        lines.append(f"  分数范围:        [{results['score_min']:.4f}, {results['score_max']:.4f}]")
        lines.append(f"  分数中位数:      {results['score_median']:.4f}")
        lines.append(f"  95分位数:        {results['score_p95']:.4f}")
        lines.append(f"  99分位数:        {results['score_p99']:.4f}")
        lines.append("=" * 50)

    return "\n".join(lines)


def find_anomaly_intervals(labels: np.ndarray, min_length: int = 1) -> list:
    """
    从二值标签数组中提取连续的异常区间

    Args:
        labels: 二值标签数组 (0=正常, 1=异常)
        min_length: 最小区间长度（短于此的区间被忽略）

    Returns:
        list of (start_idx, end_idx) 异常区间（包含端点）
    """
    labels = np.asarray(labels).flatten()
    n = len(labels)

    if n == 0:
        return []

    intervals = []
    in_interval = False
    start = 0

    for i in range(n):
        if labels[i] == 1:
            if not in_interval:
                start = i
                in_interval = True
        else:
            if in_interval:
                length = i - start
                if length >= min_length:
                    intervals.append((start, i - 1))
                in_interval = False

    # Handle last interval
    if in_interval:
        length = n - start
        if length >= min_length:
            intervals.append((start, n - 1))

    return intervals


def format_anomaly_intervals(intervals: list, scores: np.ndarray = None) -> str:
    """
    格式化异常区间为字符串

    Args:
        intervals: find_anomaly_intervals 返回的区间列表
        scores: 对应的异常分数（可选）

    Returns:
        格式化的区间描述字符串
    """
    if not intervals:
        return "未检测到连续异常区间"

    lines = []
    lines.append(f"检测到 {len(intervals)} 个连续异常区间:")
    lines.append("-" * 50)

    total_anomaly_points = sum(end - start + 1 for start, end in intervals)

    for idx, (start, end) in enumerate(intervals):
        length = end - start + 1
        if scores is not None:
            seg_scores = scores[start:end + 1]
            max_score = float(np.max(seg_scores))
            mean_score = float(np.mean(seg_scores))
            lines.append(
                f"  区间 #{idx + 1}: 索引 [{start:>5} ~ {end:>5}] | "
                f"长度: {length:>4} | "
                f"最高分: {max_score:.4f} | 均分: {mean_score:.4f}"
            )
        else:
            lines.append(f"  区间 #{idx + 1}: 索引 [{start} ~ {end}] | 长度: {length}")

    lines.append("-" * 50)
    if scores is not None:
        lines.append(f"  总异常点数: {total_anomaly_points} ({total_anomaly_points / len(scores) * 100:.2f}%)")

    return "\n".join(lines)


def evaluate_with_intervals(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    scores: np.ndarray,
    min_interval_length: int = 1
) -> Dict[str, Any]:
    """
    增强版评估：包含异常区间检测

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        scores: 异常分数
        min_interval_length: 最小连续异常区间长度

    Returns:
        包含区间信息的评估结果
    """
    results = evaluate_detector(y_true, y_pred, scores)

    # 添加区间分析
    intervals = find_anomaly_intervals(y_pred, min_length=min_interval_length)
    results['anomaly_intervals'] = intervals
    results['n_anomaly_intervals'] = len(intervals)

    if len(intervals) > 0:
        results['interval_summary'] = {
            'total_points': sum(end - start + 1 for start, end in intervals),
            'max_length': max(end - start + 1 for start, end in intervals),
            'min_length': min(end - start + 1 for start, end in intervals),
            'mean_length': np.mean([end - start + 1 for start, end in intervals]),
        }

    if y_true is not None:
        # 有标签：计算区间级别的精确率和召回率
        true_intervals = find_anomaly_intervals(y_true, min_length=min_interval_length)
        results['true_interval_count'] = len(true_intervals)

        # 区间重叠率（预测区间与真实区间的重叠程度）
        if len(true_intervals) > 0 and len(intervals) > 0:
            # 简化为：计算预测的异常点有多少比例在真实异常区间内
            true_anomaly_points = set()
            for start, end in true_intervals:
                true_anomaly_points.update(range(start, end + 1))

            pred_in_true = sum(1 for start, end in intervals
                             for i in range(start, end + 1) if i in true_anomaly_points)
            total_pred = sum(end - start + 1 for start, end in intervals)

            results['interval_precision'] = pred_in_true / max(total_pred, 1)
            results['interval_recall'] = pred_in_true / max(len(true_anomaly_points), 1)

    return results
