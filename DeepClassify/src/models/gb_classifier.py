"""
GradientBoosting 分类器 - DeepClassify
支持 XGBoost / LightGBM / sklearn GradientBoosting
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .classifier_base import ClassifierBase

logger = logging.getLogger(__name__)


class GBClassifier(ClassifierBase):
    """GradientBoosting 分类器（自动选择 XGBoost/LightGBM/sklearn）"""

    def __init__(self, **kwargs):
        super().__init__("GradientBoosting", **kwargs)
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.max_depth = kwargs.get('max_depth', 5)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.backend = kwargs.get('backend', 'auto')  # 'xgb', 'lgbm', 'sklearn', 'auto'
        self._model = None
        self._scaler: Optional[StandardScaler] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._feature_names: Optional[list] = None

    def _get_backend(self):
        """自动选择后端"""
        if self.backend != 'auto':
            return self.backend

        try:
            import xgboost
            return 'xgb'
        except ImportError:
            pass

        try:
            import lightgbm
            return 'lgbm'
        except ImportError:
            return 'sklearn'

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

            backend = self._get_backend()
            logger.info(f"GBClassifier 后端: {backend}")

            if backend == 'xgb':
                try:
                    import xgboost as xgb
                    self._model = xgb.XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        use_label_encoder=False,
                        eval_metric='mlogloss',
                        random_state=42,
                        n_jobs=-1
                    )
                except Exception:
                    import xgboost
                    self._model = xgb.XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        random_state=42,
                        n_jobs=-1
                    )
            elif backend == 'lgbm':
                import lightgbm as lgb
                self._model = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                self._model = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
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
                f"✅ GradientBoosting({backend}) 训练完成！\n\n"
                f"   类别数: {self.n_classes_}\n"
                f"   **Accuracy**: {self.metrics.get('Accuracy', 0):.4f}\n"
                f"   **F1(weighted)**: {self.metrics.get('F1_weighted', 0):.4f}\n"
                f"   **Precision**: {self.metrics.get('Precision_weighted', 0):.4f}\n"
                f"   **Recall**: {self.metrics.get('Recall_weighted', 0):.4f}"
            )
            logger.info(f"GradientBoosting 训练完成: Accuracy={self.metrics.get('Accuracy', 0):.4f}")
            return True, msg

        except Exception as e:
            logger.error(f"GradientBoosting 训练失败: {e}")
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
        if not self.is_fitted or self._model is None:
            return {}

        names = self._feature_names or []
        imp = None

        if hasattr(self._model, 'feature_importances_'):
            imp = self._model.feature_importances_
        elif hasattr(self._model, 'feature_importance_'):
            imp = self._model.feature_importance_

        if imp is not None:
            return dict(zip(names, imp))
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
        logger.info(f"GradientBoosting 模型已保存: {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self._model = data['model']
        self._scaler = data['scaler']
        self._label_encoder = data['label_encoder']
        self._feature_names = data.get('feature_names')
        self.n_classes_ = data.get('n_classes', 0)
        self.metrics = data.get('metrics', {})
        self.is_fitted = True
        logger.info(f"GradientBoosting 模型已加载: {path}")

    def get_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'backend': self._get_backend(),
        }
