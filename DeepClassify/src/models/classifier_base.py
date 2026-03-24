"""
分类器基类 - DeepClassify
所有分类器必须继承此类
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ClassifierBase(ABC):
    """分类器基类"""

    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self.feature_names: Optional[list] = None
        self.n_classes_: int = 0
        self._model: Any = None

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> tuple:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）

        Returns:
            (success: bool, message: str)
        """
        pass

    @abstractmethod
    def predict(self, X_test) -> np.ndarray:
        """预测类别标签"""
        pass

    @abstractmethod
    def predict_proba(self, X_test) -> np.ndarray:
        """预测类别概率 (n_samples, n_classes)"""
        pass

    def get_feature_importance(self) -> Dict[str, float]:
        """返回特征重要性（传统ML模型）"""
        return {}

    def save(self, path: str):
        """保存模型"""
        raise NotImplementedError(f"{self.model_type} 未实现 save 方法")

    def load(self, path: str):
        """加载模型"""
        raise NotImplementedError(f"{self.model_type} 未实现 load 方法")

    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {}

    def get_metrics_summary(self) -> str:
        """获取指标摘要"""
        if not self.metrics:
            return "未训练，无指标"
        lines = [f"{self.model_type} Metrics:"]
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return "\n".join(lines)
