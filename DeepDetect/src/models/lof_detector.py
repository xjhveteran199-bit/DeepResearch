"""
LOF (Local Outlier Factor) 异常检测器
基于密度的局部异常因子
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .detector_base import DetectorBase
import logging

logger = logging.getLogger(__name__)


class LOFDetector(DetectorBase):
    """LOF 异常检测器"""

    def __init__(self, contamination: float = 0.1, n_neighbors: int = 20,
                 metric: str = 'minkowski', p: int = 2, **kwargs):
        """
        Args:
            contamination: 异常比例
            n_neighbors: 近邻数量
            metric: 距离度量
            p: Minkowski指数
        """
        super().__init__(detector_type='LOF', contamination=contamination)

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def _build_model(self) -> LocalOutlierFactor:
        return LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            p=self.p,
            n_jobs=-1
        )

    def _fit_model(self, X_train: np.ndarray):
        # LOF 使用 novelty=False，在fit时就需要全部数据
        self.model = self._build_model()
        # LOF 需要用全部数据 fit
        predictions = self.model.fit_predict(X_train)
        # 获取训练集上的负偏移（分数越大越异常）
        self._train_scores = -self.model.negative_outlier_factor_

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """返回异常分数（LOF分数越大越异常）"""
        if not hasattr(self, '_train_scores'):
            raise RuntimeError("Model not fitted")
        # 对新数据，使用decision_function
        return -self.model.decision_function(X)

    def _get_model_state(self) -> dict:
        return {
            'model': self.model,
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'p': self.p,
            '_train_scores': getattr(self, '_train_scores', None),
        }

    def _restore_model_state(self, state: dict):
        self.model = state['model']
        self._train_scores = state.get('_train_scores')
