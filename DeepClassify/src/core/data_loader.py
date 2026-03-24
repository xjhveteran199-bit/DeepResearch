"""
数据加载模块 - DeepClassify
负责CSV导入、数据预览、自动列类型识别（数值/类别/日期）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataLoader:
    """CSV数据加载器 - 分类任务专用"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[Path] = None
        self._numeric_cols: list = []
        self._categorical_cols: list = []
        self._date_cols: list = []
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._date_parser_results: Dict[str, Any] = {}
        self._target_col: Optional[str] = None

    def load_csv(self, file_path: str) -> Tuple[bool, str]:
        """加载CSV文件，自动识别编码"""
        try:
            self.file_path = Path(file_path)
            self._label_encoders.clear()
            self._date_parser_results.clear()

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

    def _analyze_columns(self):
        """分析列类型：数值/类别/日期"""
        if self.df is None:
            return

        self._numeric_cols = []
        self._categorical_cols = []
        self._date_cols = []

        for col in self.df.columns:
            col_type = self._detect_column_type(col)
            if col_type == 'numeric':
                self._numeric_cols.append(col)
            elif col_type == 'date':
                self._date_cols.append(col)
            else:
                self._categorical_cols.append(col)

        logger.info(f"列类型识别: 数值列={len(self._numeric_cols)}, "
                    f"类别列={len(self._categorical_cols)}, 日期列={len(self._date_cols)}")

    def _detect_column_type(self, col: str) -> str:
        """智能检测列类型"""
        series = self.df[col]

        # 日期列检测
        if self._is_date_column(col, series):
            parsed = pd.to_datetime(series, errors='coerce', format='mixed')
            not_null_ratio = parsed.notna().sum() / max(len(parsed), 1)
            if not_null_ratio >= 0.8:
                self._date_parser_results[col] = parsed
                return 'date'

        # 数值列检测
        if series.dtype in ['int64', 'float64', 'int32', 'float32']:
            unique_ratio = series.nunique() / max(series.notna().sum(), 1)
            if unique_ratio > 0.1:
                return 'numeric'
            if unique_ratio <= 0.05 and series.dtype in ['int64', 'int32']:
                return 'categorical'
            return 'numeric'

        # 类别列检测
        unique_count = series.nunique()
        if unique_count <= 50 and unique_count >= 2:
            return 'categorical'
        if unique_count <= 20:
            return 'categorical'

        return 'categorical'

    def _is_date_column(self, col: str, series: pd.Series) -> bool:
        """判断是否为日期列"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        if series.dtype == 'object':
            sample = series.dropna().head(10)
            if len(sample) == 0:
                return False
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{4}/\d{2}/\d{2}',
                r'\d{2}-\d{2}-\d{4}',
                r'\d{4}年\d{1,2}月\d{1,2}日',
            ]
            import re
            for pattern in date_patterns:
                if sample.astype(str).str.match(pattern).any():
                    return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        if self.df is None:
            return {}

        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "numeric_cols": self._numeric_cols,
            "categorical_cols": self._categorical_cols,
            "date_cols": self._date_cols,
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing": self.df.isnull().sum().to_dict(),
            "numeric_stats": self.df[self._numeric_cols].describe().to_dict() if self._numeric_cols else {}
        }

    def get_preview(self, rows: int = 20) -> pd.DataFrame:
        """获取数据预览"""
        if self.df is None:
            return pd.DataFrame()
        return self.df.head(rows)

    def select_target(self, target_col: str) -> Tuple[bool, str]:
        """选择目标列"""
        if self.df is None:
            return False, "未加载数据"

        if target_col not in self.df.columns:
            return False, f"列 '{target_col}' 不存在"

        self._target_col = target_col
        unique_count = self.df[target_col].nunique()

        logger.info(f"选择目标列: {target_col}, 类别数: {unique_count}")
        return True, f"目标列: {target_col} (共 {unique_count} 个类别)"

    def get_feature_matrix(
        self,
        exclude_cols: list = None,
        return_encoded: bool = False
    ) -> pd.DataFrame:
        """获取特征矩阵（排除目标列）"""
        if self.df is None:
            return pd.DataFrame()

        exclude = set(exclude_cols or [])

        # 数值特征
        numeric_features = [c for c in self._numeric_cols if c not in exclude]

        # 类别特征 → LabelEncoder 编码
        cat_features = [c for c in self._categorical_cols if c not in exclude]
        encoded_cat = {}
        for col in cat_features:
            if col not in self._label_encoders:
                le = LabelEncoder()
                series = self.df[col].fillna(
                    self.df[col].mode()[0] if not self.df[col].mode().empty else 'missing'
                )
                encoded_cat[col] = le.fit_transform(series.astype(str))
                self._label_encoders[col] = le
            else:
                encoded_cat[col] = self._label_encoders[col].transform(
                    self.df[col].fillna('missing').astype(str)
                )

        # 日期特征 → 提取时间特征
        date_features = [c for c in self._date_cols if c not in exclude]
        for col in date_features:
            parsed = self._date_parser_results.get(col)
            if parsed is None:
                parsed = pd.to_datetime(self.df[col], errors='coerce')
            date_feats = self._extract_date_features(parsed)
            for feat_name, feat_series in date_feats.items():
                encoded_cat[feat_name] = feat_series.values

        # 合并所有特征
        all_features = {}
        if numeric_features:
            all_features.update({c: self.df[c].values for c in numeric_features})
        all_features.update(encoded_cat)

        result_df = pd.DataFrame(all_features)

        logger.info(f"特征矩阵构建完成: {result_df.shape[1]} 特征 "
                    f"(数值={len(numeric_features)}, 类别={len(cat_features)}, 日期→时间特征)")
        return result_df

    def _extract_date_features(self, parsed_series: pd.Series) -> Dict[str, pd.Series]:
        """从日期列提取时间特征"""
        features = {}
        prefix = 'date_'

        try:
            dt = pd.to_datetime(parsed_series, errors='coerce')

            features[f'{prefix}year'] = dt.dt.year.fillna(0).astype(int)
            features[f'{prefix}month'] = dt.dt.month.fillna(0).astype(int)
            features[f'{prefix}day'] = dt.dt.day.fillna(0).astype(int)
            features[f'{prefix}dayofweek'] = dt.dt.dayofweek.fillna(0).astype(int)
            features[f'{prefix}dayofyear'] = dt.dt.dayofyear.fillna(0).astype(int)
            features[f'{prefix}quarter'] = dt.dt.quarter.fillna(0).astype(int)
            features[f'{prefix}weekofyear'] = dt.dt.isocalendar().week.fillna(0).astype(int)

            if dt.dt.hour.notna().any():
                features[f'{prefix}hour'] = dt.dt.hour.fillna(0).astype(int)
                # 周期性
                features[f'{prefix}hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24).fillna(0)
                features[f'{prefix}hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24).fillna(0)

            features[f'{prefix}month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12).fillna(0)
            features[f'{prefix}month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12).fillna(0)
            features[f'{prefix}dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7).fillna(0)
            features[f'{prefix}dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7).fillna(0)

        except Exception as e:
            logger.warning(f"日期特征提取失败: {e}")

        return {k: v for k, v in features.items() if v is not None}

    def get_target(self) -> pd.Series:
        """获取目标变量（标签编码）"""
        if self.df is None or self._target_col is None:
            return pd.Series(dtype=int)

        target = self.df[self._target_col].copy()

        # 如果是数值列但类别数少，当作类别处理
        if target.dtype in ['int64', 'float64'] and target.nunique() <= 20:
            pass  # 保持原样

        return target

    def get_target_encoded(self) -> Tuple[pd.Series, LabelEncoder]:
        """获取标签编码后的目标变量"""
        if self.df is None or self._target_col is None:
            return pd.Series(dtype=int), None

        target = self.df[self._target_col].astype(str)
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(target), index=target.index)
        return y_enc, le

    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        if self.df is None or self._target_col is None:
            return {}
        return self.df[self._target_col].value_counts().to_dict()
