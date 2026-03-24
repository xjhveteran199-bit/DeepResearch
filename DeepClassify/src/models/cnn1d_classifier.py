"""
CNN1D 分类器 - DeepClassify
一维卷积神经网络，用于信号/时序分类（如心电、肌电、手势识别）
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from .classifier_base import ClassifierBase

logger = logging.getLogger(__name__)


class CNN1DClassifier(nn.Module):
    """
    CNN1D 分类网络

    架构：
    - 输入: (batch, 1, seq_len)  — 1 channel, seq_len = n_features
    - Conv1D × 2 + BatchNorm + ReLU + MaxPool
    - Global Average Pooling
    - FC → num_classes
    """

    def __init__(
        self,
        in_channels: int = 1,
        seq_len: int = 8,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes

        # 确保 seq_len >= kernel_size
        k = min(kernel_size, seq_len)

        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=k, padding=k // 2)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=k, padding=k // 2)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)

        self.pool = nn.MaxPool1d(2, 2) if seq_len >= 2 else nn.Identity()

        # 计算池化后的长度（动态）
        self._pool_len = max(1, seq_len // 2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.gap(x)           # (batch, channels, 1)
        x = x.view(x.size(0), -1)  # (batch, channels)

        out = self.fc(x)
        return out


class CNN1DClassifyWrapper(ClassifierBase):
    """CNN1D 分类器封装"""

    def __init__(self, **kwargs):
        super().__init__("CNN1D", **kwargs)
        self.hidden_channels = kwargs.get('hidden_channels', 64)
        self.kernel_size = kwargs.get('kernel_size', 3)
        self.epochs = kwargs.get('epochs', 50)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.dropout = kwargs.get('dropout', 0.3)
        self._seq_len: int = 8
        self._n_classes: int = 2
        self.device: str = "cpu"
        self._model: Optional[CNN1DClassifier] = None
        self._scaler: Optional[Any] = None
        self._label_encoder: Optional[Any] = None
        self.train_losses: list = []
        self.val_losses: list = []

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **kwargs
    ) -> Tuple[bool, str]:
        try:
            from sklearn.metrics import accuracy_score
            from sklearn.preprocessing import StandardScaler, LabelEncoder

            # 数据准备
            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train)
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)

            # 标签编码
            self._label_encoder = LabelEncoder()
            y_enc = self._label_encoder.fit_transform(y_train.astype(str))
            self.n_classes_ = len(self._label_encoder.classes_)
            self._n_classes = self.n_classes_

            # 标准化
            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train)

            # 特征数 → seq_len，padding 到至少 8
            n_features = X_train_scaled.shape[1]
            self._seq_len = max(8, n_features)

            # Padding: 如果特征数 < 8，补 0
            if n_features < self._seq_len:
                pad_width = self._seq_len - n_features
                X_train_scaled = np.pad(
                    X_train_scaled,
                    ((0, 0), (0, pad_width)),
                    mode='constant', constant_values=0
                )

            # reshape: (n_samples, 1, seq_len)
            X_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)

            # 验证集
            X_val_tensor = None
            y_val_enc = None
            if X_val is not None and y_val is not None:
                X_val_arr = np.array(X_val, dtype=np.float32)
                if X_val_arr.ndim == 1:
                    X_val_arr = X_val_arr.reshape(-1, 1)
                X_val_scaled = self._scaler.transform(X_val_arr)
                if X_val_arr.shape[1] < self._seq_len:
                    X_val_scaled = np.pad(
                        X_val_scaled,
                        ((0, 0), (0, self._seq_len - X_val_arr.shape[1])),
                        mode='constant', constant_values=0
                    )
                X_val_tensor = torch.FloatTensor(X_val_scaled).unsqueeze(1)
                y_val_enc = self._label_encoder.transform(np.array(y_val).astype(str))

            # 划分训练/验证
            split_idx = int(len(X_tensor) * 0.85)
            if X_val_tensor is None:
                X_tr, X_vl = X_tensor[:split_idx], X_tensor[split_idx:]
                y_tr_enc = y_enc[:split_idx]
                y_vl_enc = y_enc[split_idx:] if split_idx < len(y_enc) else y_enc[-1:]
            else:
                X_tr, X_vl = X_tensor, X_val_tensor
                y_tr_enc = y_enc
                y_vl_enc = y_val_enc

            # 模型
            self._model = CNN1DClassifier(
                in_channels=1,
                seq_len=self._seq_len,
                hidden_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                num_classes=self.n_classes_,
                dropout=self.dropout
            ).to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

            self.train_losses = []
            self.val_losses = []
            best_val_acc = 0.0
            best_state = None
            patience = 0

            self._model.train()
            for epoch in range(self.epochs):
                indices = torch.randperm(len(X_tr))
                epoch_loss = 0
                n_batches = 0

                for i in range(0, len(X_tr), self.batch_size):
                    batch_idx = indices[i:i + self.batch_size]
                    X_batch = X_tr[batch_idx].to(self.device)
                    y_batch = torch.LongTensor(y_tr_enc[batch_idx]).to(self.device)

                    optimizer.zero_grad()
                    pred = self._model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_train_loss = epoch_loss / max(n_batches, 1)

                # 验证
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(X_vl.to(self.device))
                    val_loss = criterion(val_pred, torch.LongTensor(y_vl_enc).to(self.device)).item()
                    val_acc = accuracy_score(
                        y_vl_enc,
                        val_pred.argmax(dim=1).cpu().numpy()
                    )
                self._model.train()

                self.train_losses.append(avg_train_loss)
                self.val_losses.append(val_loss)

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 8:
                        logger.info(f"CNN1D Early stopping at epoch {epoch + 1}")
                        break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"CNN1D Epoch {epoch + 1}/{self.epochs}, "
                                f"TrainLoss: {avg_train_loss:.4f}, ValAcc: {val_acc:.4f}")

            # 恢复最佳模型
            if best_state:
                self._model.load_state_dict(best_state)
                self._model.to(self.device)

            # 最终评估
            self._model.eval()
            with torch.no_grad():
                test_pred = self._model(X_vl.to(self.device))
                y_pred_labels = test_pred.argmax(dim=1).cpu().numpy()

            from ..core.metrics import ClassificationMetrics
            metrics_calc = ClassificationMetrics()
            y_vl_np = y_vl_enc.astype(int)
            self.metrics = metrics_calc.compute(
                y_true=y_vl_np,
                y_pred=y_pred_labels,
                labels=list(range(self.n_classes_))
            )

            self.is_fitted = True

            msg = (
                f"✅ CNN1D 训练完成！\n\n"
                f"   类别数: {self.n_classes_}\n"
                f"   特征维度: {n_features} → padding到{self._seq_len}\n"
                f"   **Accuracy**: {self.metrics.get('Accuracy', 0):.4f}\n"
                f"   **F1(weighted)**: {self.metrics.get('F1_weighted', 0):.4f}\n"
                f"   **Precision**: {self.metrics.get('Precision_weighted', 0):.4f}\n"
                f"   **Recall**: {self.metrics.get('Recall_weighted', 0):.4f}"
            )
            logger.info(f"CNN1D 训练完成: Accuracy={self.metrics.get('Accuracy', 0):.4f}")
            return True, msg

        except Exception as e:
            logger.error(f"CNN1D 训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"训练失败: {str(e)}"

    def predict(self, X_test) -> np.ndarray:
        if not self.is_fitted or self._model is None:
            raise ValueError("模型未训练，请先训练模型")

        X_arr = np.array(X_test, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        X_scaled = self._scaler.transform(X_arr)

        # Padding
        if X_arr.shape[1] < self._seq_len:
            X_scaled = np.pad(
                X_scaled,
                ((0, 0), (0, self._seq_len - X_arr.shape[1])),
                mode='constant', constant_values=0
            )

        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_tensor.to(self.device))
            pred_labels = logits.argmax(dim=1).cpu().numpy()

        if self._label_encoder:
            pred_labels = self._label_encoder.inverse_transform(pred_labels)

        return pred_labels

    def predict_proba(self, X_test) -> np.ndarray:
        if not self.is_fitted or self._model is None:
            raise ValueError("模型未训练，请先训练模型")

        X_arr = np.array(X_test, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        X_scaled = self._scaler.transform(X_arr)

        if X_arr.shape[1] < self._seq_len:
            X_scaled = np.pad(
                X_scaled,
                ((0, 0), (0, self._seq_len - X_arr.shape[1])),
                mode='constant', constant_values=0
            )

        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_tensor.to(self.device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        return proba

    def get_feature_importance(self) -> Dict[str, float]:
        """CNN 不直接支持特征重要性，返回空字典"""
        return {}

    def save(self, path: str):
        import joblib
        save_data = {
            'model_state': self._model.state_dict() if self._model else None,
            'scaler': self._scaler,
            'label_encoder': self._label_encoder,
            'seq_len': self._seq_len,
            'n_classes': self.n_classes_,
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'metrics': self.metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        joblib.dump(save_data, path)
        logger.info(f"CNN1D 模型已保存: {path}")

    def load(self, path: str):
        import joblib
        data = joblib.load(path)

        self._scaler = data['scaler']
        self._label_encoder = data['label_encoder']
        self._seq_len = data['seq_len']
        self.n_classes_ = data['n_classes']
        self._n_classes = data['n_classes']
        self.hidden_channels = data.get('hidden_channels', 64)
        self.kernel_size = data.get('kernel_size', 3)
        self.dropout = data.get('dropout', 0.3)
        self.metrics = data.get('metrics', {})
        self.train_losses = data.get('train_losses', [])
        self.val_losses = data.get('val_losses', [])

        self._model = CNN1DClassifier(
            in_channels=1,
            seq_len=self._seq_len,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            num_classes=self.n_classes_,
            dropout=self.dropout
        ).to(self.device)
        self._model.load_state_dict(data['model_state'])
        self._model.eval()

        self.is_fitted = True
        logger.info(f"CNN1D 模型已加载: {path}")

    def get_params(self) -> Dict[str, Any]:
        return {
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout,
        }
