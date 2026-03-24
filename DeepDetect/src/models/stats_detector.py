"""
统计方法异常检测器
支持 Z-score 和 IQR 两种方法
"""

import numpy as np
from .detector_base import DetectorBase
import logging

logger = logging.getLogger(__name__)


class StatsDetector(DetectorBase):
    """统计方法异常检测器"""

    def __init__(self, contamination: float = 0.1, method: str = 'zscore',
                 z_threshold: float = 3.0, iqr_factor: float = 1.5, **kwargs):
        """
        Args:
            contamination: 异常比例（仅用于自动阈值调整）
            method: 'zscore' 或 'iqr'
            z_threshold: Z-score 阈值（默认3.0，超过3个标准差为异常）
            iqr_factor: IQR倍数（默认1.5倍四分位距）
        """
        super().__init__(detector_type=f'Stats_{method.upper()}', contamination=contamination)

        self.method = method.lower()
        self.z_threshold = z_threshold
        self.iqr_factor = iqr_factor

        # 训练后存储的参数
        self._mean: np.ndarray = None
        self._std: np.ndarray = None
        self._q1: np.ndarray = None
        self._q3: np.ndarray = None
        self._iqr: np.ndarray = None

    def _fit_model(self, X_train: np.ndarray):
        """计算训练数据的统计参数"""
        self._mean = np.mean(X_train, axis=0)
        self._std = np.std(X_train, axis=0) + 1e-8  # 避免除零

        self._q1 = np.percentile(X_train, 25, axis=0)
        self._q3 = np.percentile(X_train, 75, axis=0)
        self._iqr = self._q3 - self._q1 + 1e-8  # 避免除零

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        返回异常分数
        - Z-score方法: 分数 = |(x - mean) / std|
        - IQR方法: 分数 = max(0, (x - q3) / iqr) + max(0, (q1 - x) / iqr)
        """
        if self.method == 'zscore':
            # 每个特征计算Z-score，取最大值作为综合分数
            z_scores = np.abs((X - self._mean) / self._std)
            scores = np.max(z_scores, axis=1)
        else:  # iqr
            # 计算到上下四分位的距离
            upper_dist = np.maximum(0, (X - self._q3) / self._iqr)
            lower_dist = np.maximum(0, (self._q1 - X) / self._iqr)
            # 取最大距离作为综合分数
            scores = np.maximum(upper_dist, lower_dist).max(axis=1)

        return scores

    def _get_model_state(self) -> dict:
        return {
            'method': self.method,
            'z_threshold': self.z_threshold,
            'iqr_factor': self.iqr_factor,
            '_mean': self._mean,
            '_std': self._std,
            '_q1': self._q1,
            '_q3': self._q3,
            '_iqr': self._iqr,
        }

    def _restore_model_state(self, state: dict):
        self.method = state['method']
        self.z_threshold = state['z_threshold']
        self.iqr_factor = state['iqr_factor']
        self._mean = state['_mean']
        self._std = state['_std']
        self._q1 = state['_q1']
        self._q3 = state['_q3']
        self._iqr = state['_iqr']
