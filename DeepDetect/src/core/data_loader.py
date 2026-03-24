"""
数据加载模块 - 异常检测数据加载
支持CSV导入、数据预览、自动列类型识别、缺失值处理
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
import io

logger = logging.getLogger(__name__)


class DataLoader:
    """CSV数据加载器 - 异常检测专用"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[Path] = None
        self._numeric_cols: list = []
        self._label_col: Optional[str] = None
        self._missing_info: Dict[str, Any] = {}

    def load_csv(self, file_path: str) -> Tuple[bool, str]:
        """
        加载CSV文件，自动识别编码
        Returns: (success, message)
        """
        try:
            self.file_path = Path(file_path)

            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            df = None

            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                df = pd.read_csv(file_path, encoding='utf-8', errors='replace')

            self.df = df
            self._analyze_columns()

            logger.info(f"成功加载CSV: {file_path}, 形状: {df.shape}")
            return True, f"成功加载 {len(df)} 行 × {len(df.columns)} 列数据"

        except Exception as e:
            logger.error(f"加载CSV失败: {e}")
            return False, f"加载失败: {str(e)}"

    def load_from_bytes(self, bytes_data: bytes, filename: str = "upload.csv") -> Tuple[bool, str]:
        """从字节数据加载CSV（用于Gradio文件上传）"""
        try:
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            df = None

            for enc in encodings:
                try:
                    df = pd.read_csv(io.BytesIO(bytes_data), encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8', errors='replace')

            self.df = df
            self._analyze_columns()

            logger.info(f"成功加载上传文件: {filename}, 形状: {df.shape}")
            return True, f"成功加载 {len(df)} 行 × {len(df.columns)} 列数据"

        except Exception as e:
            logger.error(f"加载上传文件失败: {e}")
            return False, f"加载失败: {str(e)}"

    def _analyze_columns(self):
        """分析数值列"""
        if self.df is None:
            return

        self._numeric_cols = []
        self._missing_info = {}

        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                self._numeric_cols.append(col)

            # 记录缺失值信息
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                self._missing_info[col] = {
                    'count': int(missing_count),
                    'ratio': float(missing_count / len(self.df))
                }

        logger.info(f"列类型识别: 数值列={len(self._numeric_cols)}, "
                    f"缺失值列={len(self._missing_info)}")

    def get_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        if self.df is None:
            return {}

        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "numeric_cols": self._numeric_cols,
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing": self.df.isnull().sum().to_dict(),
            "missing_info": self._missing_info,
            "numeric_stats": self.df[self._numeric_cols].describe().to_dict() if self._numeric_cols else {}
        }

    def get_preview(self, rows: int = 20) -> pd.DataFrame:
        """获取数据预览"""
        if self.df is None:
            return pd.DataFrame()
        return self.df.head(rows)

    def get_numeric_columns(self) -> List[str]:
        """获取所有数值列名"""
        return self._numeric_cols.copy()

    def set_label_column(self, label_col: Optional[str]):
        """设置标签列（有监督模式）"""
        if label_col and label_col not in self.df.columns:
            raise ValueError(f"Label column '{label_col}' not found")
        self._label_col = label_col

    def get_label_column(self) -> Optional[str]:
        """获取标签列名"""
        return self._label_col

    def has_labels(self) -> bool:
        """判断是否有标签"""
        return self._label_col is not None and self._label_col in self.df.columns

    def get_feature_and_label(
        self,
        target_col: str,
        handle_missing: str = 'drop'
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        获取特征矩阵和标签

        Args:
            target_col: 目标检测列
            handle_missing: 缺失值处理方式 'drop' | 'mean' | 'median'

        Returns:
            (X, y) - 特征和标签
        """
        if self.df is None:
            return pd.DataFrame(), None

        # 获取数值特征列（排除目标列）
        feature_cols = [c for c in self._numeric_cols if c != target_col]

        if not feature_cols:
            return pd.DataFrame(), None

        X = self.df[feature_cols].copy()

        # 处理缺失值
        if handle_missing == 'drop':
            mask = X.notna().all(axis=1)
            X = X[mask]
            y = self.df[self._label_col][mask] if self.has_labels() else None
        elif handle_missing == 'mean':
            X = X.fillna(X.mean())
            y = self.df[self._label_col] if self.has_labels() else None
        elif handle_missing == 'median':
            X = X.fillna(X.median())
            y = self.df[self._label_col] if self.has_labels() else None
        else:
            y = self.df[self._label_col] if self.has_labels() else None

        return X, y

    def get_all_numeric_data(
        self,
        handle_missing: str = 'drop'
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        获取全部数值数据用于无监督检测

        Returns:
            (X, y) - X为所有数值列，y为标签（如果有）
        """
        if self.df is None:
            return pd.DataFrame(), None

        X = self.df[self._numeric_cols].copy()

        if handle_missing == 'drop':
            mask = X.notna().all(axis=1)
            X = X[mask]
        elif handle_missing == 'mean':
            X = X.fillna(X.mean())
        elif handle_missing == 'median':
            X = X.fillna(X.median())

        y = None
        if self.has_labels():
            mask = X.index if handle_missing == 'drop' else None
            y = self.df[self._label_col] if mask is None else self.df[self._label_col][mask]

        return X, y

    def export_with_labels(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        scores: np.ndarray,
        anomaly_col: str = 'is_anomaly',
        score_col: str = 'anomaly_score'
    ) -> pd.DataFrame:
        """导出带异常标签的数据"""
        if self.df is None:
            return pd.DataFrame()

        # 使用原始索引对齐
        result = self.df.loc[X.index].copy()
        result[anomaly_col] = labels
        result[score_col] = scores

        return result
