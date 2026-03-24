"""
Autoencoder 异常检测器
基于 PyTorch 自编码器的重建误差检测异常
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .detector_base import DetectorBase
import logging
import io

logger = logging.getLogger(__name__)


class AutoencoderModel(nn.Module):
    """自编码器网络"""

    def __init__(self, input_dim: int, hidden_dims: list = [16, 8]):
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        for i, h_dim in enumerate(hidden_dims_rev[1:] + [input_dim]):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU() if i < len(hidden_dims_rev) - 1 else nn.Identity()
            ])
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


class AutoencoderDetector(DetectorBase):
    """Autoencoder 异常检测器"""

    def __init__(self, contamination: float = 0.1,
                 hidden_dims: list = None,
                 epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 device: str = 'auto',
                 **kwargs):
        """
        Args:
            contamination: 异常比例
            hidden_dims: 编码器隐藏层维度列表
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            device: 计算设备 'auto' | 'cuda' | 'cpu'
        """
        super().__init__(detector_type='Autoencoder', contamination=contamination)

        self.hidden_dims = hidden_dims or [16, 8]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 自动选择设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model: AutoencoderModel = None
        self._train_reconstruction_errors: np.ndarray = None

    def _fit_model(self, X_train: np.ndarray):
        """训练自编码器"""
        input_dim = X_train.shape[1]

        # 构建模型
        self.model = AutoencoderModel(input_dim, self.hidden_dims).to(self.device)

        # 准备数据
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

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

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"[Autoencoder] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

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
