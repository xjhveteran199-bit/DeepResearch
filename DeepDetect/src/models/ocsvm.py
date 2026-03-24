"""
One-Class SVM 异常检测器
"""

import numpy as np
from sklearn.svm import OneClassSVM
from .detector_base import DetectorBase
import logging

logger = logging.getLogger(__name__)


class OCSVMDetector(DetectorBase):
    """One-Class SVM 异常检测器"""

    def __init__(self, contamination: float = 0.1, kernel: str = 'rbf',
                 nu: float = 0.1, gamma: str = 'scale', **kwargs):
        """
        Args:
            contamination: 异常比例
            kernel: 核函数类型
            nu: nu参数（类似contamination）
            gamma: RBF核参数
        """
        super().__init__(detector_type='OneClassSVM', contamination=contamination)

        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma

    def _build_model(self) -> OneClassSVM:
        return OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        )

    def _fit_model(self, X_train: np.ndarray):
        self.model = self._build_model()
        self.model.fit(X_train)

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        返回异常分数（sklearn OCSVM的decision_function返回负值，越小越异常）
        我们取负值使分数越大越异常
        """
        raw_scores = self.model.decision_function(X)
        return -raw_scores

    def _get_model_state(self) -> dict:
        return {
            'model': self.model,
            'kernel': self.kernel,
            'nu': self.nu,
            'gamma': self.gamma,
        }

    def _restore_model_state(self, state: dict):
        self.model = state['model']
