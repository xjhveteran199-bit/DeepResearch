"""
LSTM-based 异常检测器
基于 PyTorch LSTM 的重建误差检测时序异常
思路：训练 LSTM 学习时序模式，异常点偏离预测值越多，分数越高
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .detector_base import DetectorBase
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class LSTMPredictorModel(nn.Module):
    """LSTM 预测模型：输入 seq_len 个历史值，预测下一个值"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, input_size)  # 预测所有特征
        )

        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMDetector(DetectorBase):
    """
    LSTM 时序异常检测器

    训练阶段：用正常数据学习时序模式
    检测阶段：计算预测误差作为异常分数，误差越大越异常

    适用于：单变量或多变量时序数据的异常检测
    """

    def __init__(self, contamination: float = 0.1,
                 seq_len: int = 20,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 epochs: int = 50,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 device: str = 'auto',
                 **kwargs):
        """
        Args:
            contamination: 异常比例
            seq_len: 滑动窗口长度（时序历史步数）
            hidden_size: LSTM 隐藏层大小
            num_layers: LSTM 层数
            dropout: Dropout 比例
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            device: 计算设备 'auto' | 'cuda' | 'cpu'
        """
        super().__init__(detector_type='LSTM', contamination=contamination)

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model: LSTMPredictorModel = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.input_size: int = 1
        self.best_state: Optional[dict] = None

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建滑动窗口序列"""
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i + self.seq_len])
            y.append(data[i + self.seq_len])
        return np.array(X), np.array(y)

    def _normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std, mean, std

    def _fit_model(self, X_train: np.ndarray):
        """训练 LSTM 模型"""
        self.input_size = X_train.shape[1] if X_train.ndim > 1 else 1
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        # 归一化
        X_norm, self.scaler_mean, self.scaler_std = self._normalize(X_train)

        # 创建序列
        X_seq, y_seq = self._create_sequences(X_norm)
        # Keep y_seq as (n_samples, n_features) - multi-output prediction
        y_seq = np.asarray(y_seq)

        # 划分训练/验证
        split_idx = max(1, int(len(X_seq) * 0.85))
        X_tr, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_tr, y_val = y_seq[:split_idx], y_seq[split_idx:]

        X_tr_t = torch.FloatTensor(X_tr).to(self.device)
        y_tr_t = torch.FloatTensor(y_tr).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 构建模型
        self.model = LSTMPredictorModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        best_val_loss = float('inf')
        patience_counter = 0
        MAX_PATIENCE = 10

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_X)  # (B, n_features)
                loss = criterion(output, batch_y)  # MSE between predicted and actual next step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t).cpu().numpy()  # (N_val, n_features)
                # Compute per-feature MSE and average
                val_mse = float(np.mean((val_preds - y_val) ** 2))
            self.model.train()

            scheduler.step(val_mse)

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"[LSTM] Epoch {epoch+1}/{self.epochs}, "
                    f"TrainLoss: {avg_train_loss:.6f}, ValMSE: {val_mse:.6f}, "
                    f"Patience: {patience_counter}"
                )

            if patience_counter >= MAX_PATIENCE:
                logger.info(f"[LSTM] Early stopping at epoch {epoch+1}")
                break

        # 恢复最佳模型
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        self.model.eval()
        logger.info(f"[LSTM] 训练完成，最佳 ValMSE: {best_val_loss:.6f}")

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """计算预测误差作为异常分数"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = len(X)
        # 初始化分数数组（与输入同长，但前 seq_len 个位置设为0）
        scores = np.zeros(n)

        # 归一化
        X_norm = (X - self.scaler_mean) / self.scaler_std

        # 逐点预测并计算误差
        self.model.eval()
        with torch.no_grad():
            for i in range(self.seq_len, n):
                seq = torch.FloatTensor(X_norm[i - self.seq_len:i]).unsqueeze(0).to(self.device)
                pred = self.model(seq).cpu().numpy().ravel()
                # 预测值
                pred_denorm = pred * self.scaler_std + self.scaler_mean
                # 实际值
                actual = X[i]
                # MSE 作为异常分数
                error = np.mean((pred_denorm - actual) ** 2)
                scores[i] = error

        return scores

    def _get_model_state(self) -> dict:
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'seq_len': self.seq_len,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'device': self.device,
        }

    def _restore_model_state(self, state: dict):
        self.input_size = state['input_size']
        self.hidden_size = state['hidden_size']
        self.num_layers = state['num_layers']
        self.seq_len = state['seq_len']
        self.scaler_mean = state['scaler_mean']
        self.scaler_std = state['scaler_std']
        self.device = state['device']
        self.model = LSTMPredictorModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
