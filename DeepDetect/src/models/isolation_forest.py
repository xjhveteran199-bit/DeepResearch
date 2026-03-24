"""
IsolationForest 异常检测器
基于随机森林的隔离方法
"""

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF
from .detector_base import DetectorBase
import logging

logger = logging.getLogger(__name__)


class IsolationForestDetector(DetectorBase):
    """IsolationForest 异常检测器"""

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100,
                 max_samples: str = 'auto', random_state: int = 42, **kwargs):
        """
        Args:
            contamination: 异常比例
            n_estimators: 树的数量
            max_samples: 每个样本的最大采样数
            random_state: 随机种子
        """
        super().__init__(detector_type='IsolationForest', contamination=contamination)

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def _build_model(self) -> SklearnIF:
        """构建 sklearn IsolationForest 模型"""
        return SklearnIF(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )

    def _fit_model(self, X_train: np.ndarray):
        self.model = self._build_model()
        # sklearn IF 用全部数据训练，predict时自动考虑contamination
        self.model.fit(X_train)

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        返回异常分数（sklearn IF的score_samples返回负值，越小越异常）
        我们取负值使分数越大越异常
        """
        raw_scores = self.model.score_samples(X)
        # 转换为：越大越异常
        return -raw_scores

    def _get_model_state(self) -> dict:
        return {
            'model': self.model,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
        }

    def _restore_model_state(self, state: dict):
        self.model = state['model']
