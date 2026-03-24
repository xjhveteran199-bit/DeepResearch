"""
检测器基类 - 所有异常检测器的父类
定义统一的接口：fit / predict / score_samples / save / load
"""

import numpy as np
import pickle
from abc import ABC, abstractmethod
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class DetectorBase(ABC):
    """
    异常检测器基类

    所有检测器必须实现以下方法：
    - _build_model(): 构建内部模型
    - _fit_model(X_train): 训练模型
    - _predict_scores(X): 计算异常分数
    """

    def __init__(self, detector_type: str, contamination: float = 0.1, **kwargs):
        """
        Args:
            detector_type: 检测器类型名称
            contamination: 预期的异常比例 (0.0 ~ 1.0)
        """
        self.detector_type = detector_type
        self.contamination = contamination
        self.model: Optional[Any] = None
        self.threshold: Optional[float] = None
        self._is_fitted = False
        self._feature_names: Optional[list] = None
        self._score_stats: Optional[dict] = None

    def fit(self, X_train: np.ndarray) -> 'DetectorBase':
        """
        训练检测器（无监督：只用正常数据训练）

        Args:
            X_train: 训练数据 (n_samples, n_features)

        Returns:
            self
        """
        X_train = self._validate_input(X_train)

        logger.info(f"[{self.detector_type}] 开始训练，样本数: {X_train.shape[0]}, 特征数: {X_train.shape[1]}")

        # 子类实现训练逻辑
        self._fit_model(X_train)

        # 计算训练集上的分数统计，用于确定阈值
        train_scores = self._predict_scores(X_train)
        self._score_stats = {
            'mean': float(np.mean(train_scores)),
            'std': float(np.std(train_scores)),
            'min': float(np.min(train_scores)),
            'max': float(np.max(train_scores)),
            'p95': float(np.percentile(train_scores, 95)),
            'p99': float(np.percentile(train_scores, 99)),
        }

        # 根据contamination设置阈值
        self.threshold = float(np.percentile(train_scores, (1 - self.contamination) * 100))

        self._is_fitted = True
        self._feature_names = None

        logger.info(f"[{self.detector_type}] 训练完成，阈值: {self.threshold:.4f}")
        return self

    @abstractmethod
    def _fit_model(self, X_train: np.ndarray):
        """子类实现具体的训练逻辑"""
        pass

    @abstractmethod
    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """子类实现计算异常分数"""
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        预测异常标签

        Args:
            X_test: 测试数据 (n_samples, n_features)

        Returns:
            标签数组 (0=正常, 1=异常)
        """
        if not self._is_fitted:
            raise RuntimeError(f"[{self.detector_type}] 模型未训练，请先调用 fit()")

        X_test = self._validate_input(X_test)
        scores = self._predict_scores(X_test)
        predictions = (scores > self.threshold).astype(int)

        n_anomalies = int(np.sum(predictions))
        logger.info(f"[{self.detector_type}] 预测完成，异常数: {n_anomalies}/{len(predictions)}")

        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        返回异常分数（分数越高越异常）

        Args:
            X: 数据 (n_samples, n_features)

        Returns:
            异常分数数组
        """
        if not self._is_fitted:
            raise RuntimeError(f"[{self.detector_type}] 模型未训练，请先调用 fit()")

        X = self._validate_input(X)
        return self._predict_scores(X)

    def get_threshold(self) -> float:
        """返回判定阈值"""
        if self.threshold is None:
            raise RuntimeError(f"[{self.detector_type}] 阈值未设置，请先调用 fit()")
        return self.threshold

    def get_score_stats(self) -> dict:
        """返回训练集上的分数统计"""
        return self._score_stats or {}

    def save(self, path: str):
        """
        保存模型到文件

        Args:
            path: 保存路径
        """
        state = {
            'detector_type': self.detector_type,
            'contamination': self.contamination,
            'threshold': self.threshold,
            '_is_fitted': self._is_fitted,
            '_score_stats': self._score_stats,
            'model_state': self._get_model_state(),
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"[{self.detector_type}] 模型已保存: {path}")

    def load(self, path: str):
        """
        从文件加载模型

        Args:
            path: 模型路径
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.detector_type = state['detector_type']
        self.contamination = state['contamination']
        self.threshold = state['threshold']
        self._is_fitted = state['_is_fitted']
        self._score_stats = state['_score_stats']

        self._restore_model_state(state['model_state'])

        logger.info(f"[{self.detector_type}] 模型已加载: {path}")

    @abstractmethod
    def _get_model_state(self) -> dict:
        """子类实现获取模型状态"""
        pass

    @abstractmethod
    def _restore_model_state(self, state: dict):
        """子类实现恢复模型状态"""
        pass

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """验证和标准化输入"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.detector_type}, contamination={self.contamination})"
