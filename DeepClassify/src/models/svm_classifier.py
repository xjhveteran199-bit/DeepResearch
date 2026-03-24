"""
SVM 分类器 - DeepClassify
支持向量机分类器
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .classifier_base import ClassifierBase

logger = logging.getLogger(__name__)


class SVMClassifier(ClassifierBase):
    """SVM 分类器"""

    def __init__(self, **kwargs):
        super().__init__("SVM", **kwargs)
        self.C = kwargs.get('C', 1.0)
        self.kernel = kwargs.get('kernel', 'rbf')
        self.gamma = kwargs.get('gamma', 'scale')
        self._model: Optional[SVC] = None
        self._scaler: Optional[StandardScaler] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._feature_names: Optional[list] = None

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **kwargs
    ) -> Tuple[bool, str]:
        try:
            from ..core.metrics import ClassificationMetrics

            X_arr = np.array(X_train, dtype=np.float32)
            y_arr = np.array(y_train)

            self._label_encoder = LabelEncoder()
            y_enc = self._label_encoder.fit_transform(y_arr.astype(str))
            self.n_classes_ = len(self._label_encoder.classes_)

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_arr)

            self._feature_names = (
                list(X_train.columns) if hasattr(X_train, 'columns')
                else [f"f{i}" for i in range(X_scaled.shape[1])]
            )

            self._model = SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                probability=True,
                random_state=42
            )
            self._model.fit(X_scaled, y_enc)

            y_pred = self._model.predict(X_scaled)
            metrics_calc = ClassificationMetrics()
            self.metrics = metrics_calc.compute(
                y_true=y_enc,
                y_pred=y_pred,
                labels=list(range(self.n_classes_))
            )

            self.is_fitted = True

            msg = (
                f"✅ SVM 训练完成！\n\n"
                f"   类别数: {self.n_classes_}\n"
                f"   kernel: {self.kernel}\n"
                f"   **Accuracy**: {self.metrics.get('Accuracy', 0):.4f}\n"
                f"   **F1(weighted)**: {self.metrics.get('F1_weighted', 0):.4f}\n"
                f"   **Precision**: {self.metrics.get('Precision_weighted', 0):.4f}\n"
                f"   **Recall**: {self.metrics.get('Recall_weighted', 0):.4f}"
            )
            logger.info(f"SVM 训练完成: Accuracy={self.metrics.get('Accuracy', 0):.4f}")
            return True, msg

        except Exception as e:
            logger.error(f"SVM 训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"训练失败: {str(e)}"

    def predict(self, X_test) -> np.ndarray:
        if not self.is_fitted or self._model is None:
            raise ValueError("模型未训练，请先训练模型")

        X_arr = np.array(X_test, dtype=np.float32)
        X_scaled = self._scaler.transform(X_arr)
        pred_enc = self._model.predict(X_scaled)

        return self._label_encoder.inverse_transform(pred_enc)

    def predict_proba(self, X_test) -> np.ndarray:
        if not self.is_fitted or self._model is None:
            raise ValueError("模型未训练，请先训练模型")

        X_arr = np.array(X_test, dtype=np.float32)
        X_scaled = self._scaler.transform(X_arr)
        return self._model.predict_proba(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """SVM 不直接支持特征重要性（核方法），返回空"""
        return {}

    def save(self, path: str):
        import joblib
        joblib.dump({
            'model': self._model,
            'scaler': self._scaler,
            'label_encoder': self._label_encoder,
            'feature_names': self._feature_names,
            'n_classes': self.n_classes_,
            'metrics': self.metrics,
        }, path)
        logger.info(f"SVM 模型已保存: {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self._model = data['model']
        self._scaler = data['scaler']
        self._label_encoder = data['label_encoder']
        self._feature_names = data.get('feature_names')
        self.n_classes_ = data.get('n_classes', 0)
        self.metrics = data.get('metrics', {})
        self.is_fitted = True
        logger.info(f"SVM 模型已加载: {path}")

    def get_params(self) -> Dict[str, Any]:
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
        }
