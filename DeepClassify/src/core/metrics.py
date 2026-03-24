"""
分类评估指标模块 - DeepClassify
提供 Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, PR-AUC 等指标
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    roc_curve, auc
)

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """分类评估指标计算器"""

    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None
        self._y_proba: Optional[np.ndarray] = None
        self._class_labels: Optional[list] = None

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        labels: list = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        计算所有分类指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率 (n_samples, n_classes)
            labels: 类别标签列表
            average: 平均方式 ('micro', 'macro', 'weighted')
        """
        self._y_true = np.array(y_true).ravel()
        self._y_pred = np.array(y_pred).ravel()
        self._y_proba = y_proba
        self._class_labels = labels

        self.metrics = {}

        # Accuracy
        self.metrics['Accuracy'] = accuracy_score(y_true, y_pred)

        # Precision
        self.metrics['Precision_micro'] = precision_score(
            y_true, y_pred, average='micro', zero_division=0
        )
        self.metrics['Precision_macro'] = precision_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        self.metrics['Precision_weighted'] = precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Recall
        self.metrics['Recall_micro'] = recall_score(
            y_true, y_pred, average='micro', zero_division=0
        )
        self.metrics['Recall_macro'] = recall_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        self.metrics['Recall_weighted'] = recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # F1-score
        self.metrics['F1_micro'] = f1_score(
            y_true, y_pred, average='micro', zero_division=0
        )
        self.metrics['F1_macro'] = f1_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        self.metrics['F1_weighted'] = f1_score(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # ROC-AUC / PR-AUC (需要概率)
        if y_proba is not None and len(y_proba.shape) == 2:
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                # 二分类
                self.metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba[:, 1])
                self.metrics['PR_AUC'] = average_precision_score(y_true, y_proba[:, 1])
            else:
                # 多分类
                try:
                    self.metrics['ROC_AUC_micro'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average=average
                    )
                    self.metrics['PR_AUC_macro'] = average_precision_score(
                        y_true, y_proba, average='macro'
                    )
                except Exception as e:
                    logger.warning(f"ROC-AUC/PR-AUC 计算失败: {e}")

        logger.info(f"指标计算完成: Accuracy={self.metrics['Accuracy']:.4f}")
        return self.metrics

    def get_confusion_matrix(self, normalize: bool = False) -> Tuple[np.ndarray, list]:
        """获取混淆矩阵"""
        if self._y_true is None or self._y_pred is None:
            return np.array([]), []

        cm = confusion_matrix(self._y_true, self._y_pred)
        if normalize:
            cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
            cm = np.nan_to_num(cm)

        labels = self._class_labels or sorted(set(self._y_true.tolist()))
        return cm, labels

    def get_classification_report(self) -> str:
        """获取文本分类报告"""
        if self._y_true is None or self._y_pred is None:
            return ""

        labels = self._class_labels if self._class_labels else np.unique(np.concatenate([self._y_true, self._y_pred]))
        target_names = [f"Class_{l}" for l in labels]

        return classification_report(
            y_true=self._y_true,
            y_pred=self._y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        )

    def get_roc_curve_data(self) -> Dict[int, Dict[str, np.ndarray]]:
        """获取每个类别的 ROC 曲线数据"""
        if self._y_true is None or self._y_proba is None:
            return {}

        if len(self._y_proba.shape) != 2:
            return {}

        n_classes = self._y_proba.shape[1]
        roc_data = {}

        for i in range(n_classes):
            y_true_bin = (self._y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, self._y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[i] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

        return roc_data

    def get_summary(self) -> str:
        """获取指标摘要文本"""
        if not self.metrics:
            return "未计算指标"

        lines = []
        lines.append("=" * 40)
        lines.append("       Classification Metrics Summary")
        lines.append("=" * 40)
        lines.append(f"  Accuracy:      {self.metrics.get('Accuracy', 0):.4f}")
        lines.append(f"  Precision(w): {self.metrics.get('Precision_weighted', 0):.4f}")
        lines.append(f"  Recall(w):     {self.metrics.get('Recall_weighted', 0):.4f}")
        lines.append(f"  F1-score(w):   {self.metrics.get('F1_weighted', 0):.4f}")
        if 'ROC_AUC' in self.metrics:
            lines.append(f"  ROC-AUC:       {self.metrics.get('ROC_AUC', 0):.4f}")
        if 'PR_AUC' in self.metrics:
            lines.append(f"  PR-AUC:        {self.metrics.get('PR_AUC', 0):.4f}")
        lines.append("=" * 40)
        return "\n".join(lines)
