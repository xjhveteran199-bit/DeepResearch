"""
Autoencoder 异常检测器
基于 PyTorch 自编码器的重建误差检测异常
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .detector_base import DetectorBase
import logging

logger = logging.getLogger(__name__)


class AutoencoderModel(nn.Module):
    """自编码器网络（带 Batch Normalization）"""

    def __init__(self, input_dim: int, hidden_dims: list = [16, 8]):
        super().__init__()
        self.input_dim = input_dim

        # Encoder: Linear -> BatchNorm -> ReLU
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: Linear -> BatchNorm -> ReLU (最后一层不用激活)
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        for i, h_dim in enumerate(hidden_dims_rev[1:] + [input_dim]):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            if i < len(hidden_dims_rev) - 1:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
                decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class EarlyStopping:
    """早停：监控 loss，patience 个 epoch 没有改善则停止"""
    def __init__(self, patience: int = 5, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class AutoencoderDetector(DetectorBase):
    """Autoencoder 异常检测器"""

    def __init__(self, contamination: float = 0.1,
                 hidden_dims: list = None,
                 epochs: int = 50,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 device: str = 'auto',
                 early_stopping_patience: int = 5,
                 **kwargs):
        """
        Args:
            contamination: 异常比例
            hidden_dims: 编码器隐藏层维度列表
            epochs: 训练轮数（默认50，早停可提前结束）
            batch_size: 批大小
            learning_rate: 学习率
            device: 计算设备 'auto' | 'cuda' | 'cpu'
            early_stopping_patience: 早停耐心值（多少 epoch 没改善则停止）
        """
        super().__init__(detector_type='Autoencoder', contamination=contamination)

        self.hidden_dims = hidden_dims or [16, 8]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        # 自动选择设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model: AutoencoderModel = None
        self._train_reconstruction_errors: np.ndarray = None

    def _fit_model(self, X_train: np.ndarray):
        """训练自编码器（带早停 + LR调度）"""
        input_dim = X_train.shape[1]

        # 构建模型
        self.model = AutoencoderModel(input_dim, self.hidden_dims).to(self.device)

        # 准备数据
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # LR 调度：loss 平台期时降低学习率
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
        criterion = nn.MSELoss()

        # 早停
        early_stopping = EarlyStopping(patience=self.early_stopping_patience)

        # 训练
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']

            if (epoch + 1) % 10 == 0 or early_stopping.early_stop:
                logger.info(f"[Autoencoder] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

            early_stopping(avg_loss)
            if early_stopping.early_stop:
                logger.info(f"[Autoencoder] Early stopping at epoch {epoch+1} (no improvement for {self.early_stopping_patience} epochs)")
                break

        # 计算训练集重建误差
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_train).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1)
            self._train_reconstruction_errors = errors.cpu().numpy()

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """计算重建误差作为异常分数"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1)
            return errors.cpu().numpy()

    def _get_model_state(self) -> dict:
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'input_dim': self.model.input_dim if self.model else None,
            'hidden_dims': self.hidden_dims,
            'device': self.device,
            '_train_reconstruction_errors': self._train_reconstruction_errors,
        }

    def _restore_model_state(self, state: dict):
        self.hidden_dims = state['hidden_dims']
        self.device = state['device']
        self.model = AutoencoderModel(state['input_dim'], self.hidden_dims).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self._train_reconstruction_errors = state['_train_reconstruction_errors']

    def save(self, path: str):
        """保存模型（包含PyTorch模型权重）"""
        state = {
            'detector_type': self.detector_type,
            'contamination': self.contamination,
            'threshold': self.threshold,
            '_is_fitted': self._is_fitted,
            '_score_stats': self._score_stats,
            '_model_state': self._get_model_state(),
        }

        with open(path, 'wb') as f:
            torch.save(state, f)

        logger.info(f"[{self.detector_type}] 模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            state = torch.load(f, map_location=self.device, weights_only=False)

        self.detector_type = state['detector_type']
        self.contamination = state['contamination']
        self.threshold = state['threshold']
        self._is_fitted = state['_is_fitted']
        self._score_stats = state['_score_stats']

        model_state = state['_model_state']
        self.hidden_dims = model_state['hidden_dims']
        self.device = model_state['device']
        self.model = AutoencoderModel(model_state['input_dim'], self.hidden_dims).to(self.device)
        self.model.load_state_dict(model_state['model_state_dict'])
        self._train_reconstruction_errors = model_state['_train_reconstruction_errors']

        logger.info(f"[{self.detector_type}] 模型已加载: {path}")
