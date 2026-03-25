"""
DeepPredict 独立网站 v2.0
面向研究者的时序预测工具，支持 LSTM / CNN1D / PatchTST / RF / GB / Linear / Logistic

升级日志 v2.0:
- [修复] LogisticRegression P0 静默替换 → 显式报错
- [修复] Date列 fillna 崩溃 → 只对数值列操作
- [新增] SHAP 分析自动运行并打包进 ZIP
- [升级] 所有图表 300 DPI，Nature 配色
"""

import os
import time as _time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "7861"

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
import json
import zipfile
import io
import uuid
import tempfile
import warnings
warnings.filterwarnings('ignore')

# ===== 路径配置 =====
DEEP_PREDICT_ROOT = Path(r"C:\Users\XJH\DeepResearch\DeepPredict")
DEEP_PREDICT_SRC = DEEP_PREDICT_ROOT / "src"
sys.path.insert(0, str(DEEP_PREDICT_SRC))
sys.path.insert(0, str(DEEP_PREDICT_ROOT))

# ===== matplotlib Nature 风格配置 =====
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['text.usetex'] = False

# ===== 日志 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ===== Nature 配色 =====
NATURE_PALETTE = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
    '#F39B7F', '#8491B4', '#91D1C2', '#DC000C',
    '#7E6148', '#B09C85', '#F0E442', '#55A1B1'
]

# ============================================================
# ========== DeepPredict 数据加载器 =========================
# ============================================================

class DPDataLoader:
    def __init__(self):
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.date_cols = []
        self.all_cols = []

    def load(self, file_path):
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        for enc in encodings:
            try:
                self.df = pd.read_csv(file_path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if self.df is None:
            self.df = pd.read_csv(file_path, encoding='utf-8', errors='replace')

        self.all_cols = list(self.df.columns)
        self.numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        self.date_cols = [c for c in self.all_cols if c.lower() in
                         {'timestamp','date','datetime','time'} or
                         pd.api.types.is_datetime64_any_dtype(self.df[c])]
        self.categorical_cols = [c for c in self.all_cols
                                if c not in self.numeric_cols and c not in self.date_cols]
        return len(self.df)

    def get_info(self):
        if self.df is None:
            return ""
        info = f"**数据形状**：{self.df.shape[0]} 行 × {self.df.shape[1]} 列\n\n"
        if self.numeric_cols:
            num_details = []
            for col in self.numeric_cols[:8]:
                col_data = self.df[col].dropna()
                if len(col_data) > 0:
                    vmin = f"{col_data.min():.4g}"
                    vmax = f"{col_data.max():.4g}"
                    if vmin != vmax:
                        num_details.append(f"`{col}` [{vmin} ~ {vmax}]")
                    else:
                        num_details.append(f"`{col}` ={vmin}")
            info += f"**数值列**（{len(self.numeric_cols)}）：" + "，".join(num_details)
            if len(self.numeric_cols) > 8:
                info += f"...（共{len(self.numeric_cols)}列）"
            info += "\n\n"
        info += f"**类别列**（{len(self.categorical_cols)}）：`{'`，`'.join(self.categorical_cols[:5])}{'...' if len(self.categorical_cols)>5 else ''}`"
        return info

    def get_feature_matrix(self, exclude_cols=None):
        exclude = set(exclude_cols or [])
        cols = [c for c in self.all_cols if c not in exclude]
        return self.df[cols]


# ============================================================
# ========== DeepPredict 预测器 ==============================
# ============================================================

class DPPredictor:
    def __init__(self):
        self.is_fitted = False
        self.model = None
        self.task_type = None
        self.feature_names = []
        self.metrics = {}
        self._is_lstm = False
        self._is_patchtst = False
        self._lstm_model = None
        self.shap_figures = {}
        self._scaler = None

    def train(self, X_df, y_series, target_col, model_name, params, test_size=0.2):
        import logging
        logger = logging.getLogger(__name__)

        # 数据准备
        self.feature_names = list(X_df.columns)
        numeric_mask = X_df.select_dtypes(include=[np.number]).columns
        X = X_df[numeric_mask].fillna(X_df[numeric_mask].median())
        if X.shape[1] == 0:
            return False, "❌ 选中特征中没有任何数值列，无法进行回归/分类", {}

        self.task_type = 'classification' if model_name in ('LogisticRegression',) else 'regression'
        X_arr = X.values
        y_arr = y_series.values

        # P0：LogisticRegression 不能用于回归
        if model_name == 'LogisticRegression' and self.task_type == 'regression':
            return False, "❌ LogisticRegression 是分类器，不能用于回归任务。请选择 LinearRegression、Ridge 或 ElasticNet。", {}

        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X_arr, y_arr, test_size=float(test_size), random_state=42)

        if model_name == 'LSTM':
            self._is_lstm = True
            X_arr_lstm = X.values.astype(np.float32)
            y_arr_lstm = y_series.values.astype(np.float32)
            X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_arr_lstm, y_arr_lstm, test_size=float(test_size), random_state=42)
            try:
                from DeepPredict.src.models.lstm_model import LSTMPredictor
                self._lstm_model = LSTMPredictor()
                self._lstm_model.fit(X_tr2, y_tr2, **params)
                preds = self._lstm_model.predict(X_te2)
                train_loss = self._lstm_model.get_training_loss()
                msg = f"✅ LSTM 训练完成！Loss: {train_loss[-1]:.4f}" if train_loss else "✅ LSTM 训练完成"
                self.metrics = {}
            except Exception as e:
                return False, f"❌ LSTM 训练失败: {e}", {}
            self.is_fitted = True
            return True, msg, {}

        elif model_name == 'PatchTST':
            self._is_patchtst = True
            X_arr_pt = X.values.astype(np.float32)
            y_arr_pt = y_series.values.astype(np.float32)
            X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_arr_pt, y_arr_pt, test_size=float(test_size), random_state=42)
            try:
                from DeepPredict.src.models.patchtst_model import PatchTSTPredictor
                self._lstm_model = PatchTSTPredictor()
                self._lstm_model.fit(X_tr2, y_tr2, **params)
                preds = self._lstm_model.predict(X_te2)
                msg = "✅ PatchTST 训练完成"
                self.metrics = {}
            except Exception as e:
                return False, f"❌ PatchTST 训练失败: {e}", {}
            self.is_fitted = True
            return True, msg, {}

        else:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
            from sklearn.linear_model import LinearRegression, LogisticRegression

            model_map = {
                ('RandomForest', 'regression'): (RandomForestRegressor, {}),
                ('RandomForest', 'classification'): (RandomForestClassifier, {}),
                ('GradientBoosting', 'regression'): (GradientBoostingRegressor, {}),
                ('GradientBoosting', 'classification'): (GradientBoostingClassifier, {}),
                ('LinearRegression', 'regression'): (LinearRegression, {}),
                ('LogisticRegression', 'classification'): (LogisticRegression, {}),
            }
            key = (model_name, self.task_type)
            model_cls = model_map.get(key)
            if model_cls is None:
                return False, f"❌ 不支持的模型: {model_name} ({self.task_type})", {}
            model_cls, extra_params = model_cls
            clf = model_cls(**extra_params)

            try:
                clf.fit(X_tr, y_tr)
            except Exception as e:
                return False, f"❌ 训练失败: {e}", {}

            preds = clf.predict(X_te)
            self.model = clf
            self.metrics = {}

            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            if self.task_type == 'regression':
                r2 = r2_score(y_te, preds)
                rmse = np.sqrt(mean_squared_error(y_te, preds))
                mae = mean_absolute_error(y_te, preds)
                self.metrics = {'R²': round(r2, 4), 'RMSE': round(rmse, 4), 'MAE': round(mae, 4)}
                msg = f"✅ {model_name} 训练完成  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}"
            else:
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(y_te, preds)
                self.metrics = {'Accuracy': round(acc, 4)}
                msg = f"✅ {model_name} 训练完成  Accuracy={acc:.4f}"

            self.is_fitted = True
            return True, msg, {}

    def predict(self, X_df):
        if not self.is_fitted:
            raise ValueError("模型未训练")
        numeric_mask = X_df.select_dtypes(include=[np.number]).columns
        X = X_df[numeric_mask].fillna(X_df[numeric_mask].median())
        if self._is_lstm or self._is_patchtst:
            X_arr = X.values.astype(np.float32)
            return self._lstm_model.predict(X_arr)
        if self.model is None:
            raise ValueError("模型未正确训练，无法预测")
        return self.model.predict(X)

    def run_shap_analysis(self, X_df):
        """运行 SHAP 分析并保存图表"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}, "SHAP 仅支持 sklearn 树模型"
        numeric_mask = X_df.select_dtypes(include=[np.number]).columns
        X = X_df[numeric_mask].fillna(X_df[numeric_mask].median())
        try:
            import shap
        except ImportError:
            return {}, "⚠️ SHAP 库未安装"
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            fig, ax = plt.subplots(figsize=(8, 5))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], X, plot_type="dot", show=False, max_display=20)
            else:
                shap.summary_plot(shap_values, X, plot_type="dot", show=False, max_display=20)
            fig.tight_layout()
            imp_path = os.path.join(tempfile.gettempdir(), f"shap_importance_{uuid.uuid4().hex[:8]}.png")
            fig.savefig(imp_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            fig_beeswarm = None
            try:
                fig_beeswarm, _ = plt.subplots()
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values[0], X, plot_type="layered_violin", show=False, max_display=20)
                else:
                    shap.summary_plot(shap_values, X, plot_type="layered_violin", show=False, max_display=20)
                fig_beeswarm.tight_layout()
                bsw_path = os.path.join(tempfile.gettempdir(), f"shap_beeswarm_{uuid.uuid4().hex[:8]}.png")
                fig_beeswarm.savefig(bsw_path, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig_beeswarm)
                self.shap_figures = {'importance': imp_path, 'beeswarm': bsw_path}
            except Exception:
                self.shap_figures = {'importance': imp_path}
            return self.shap_figures, "✅ SHAP 分析完成"
        except Exception as e:
            return {}, f"⚠️ SHAP 分析失败: {e}"

    def download_package(self, X_df, y_series, preds):
        """生成完整 ZIP 包"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 预测数据 CSV（完整全量数据）
            full_df = X_df.copy()
            full_df['target_actual'] = y_series.values
            full_df['target_predicted'] = preds
            csv_buf = io.StringIO()
            full_df.to_csv(csv_buf, index=False, encoding='utf-8')
            zf.writestr('forecast_data.csv', csv_buf.getvalue())

            # metrics JSON
            if self.metrics:
                zf.writestr('metrics.json', json.dumps(self.metrics, indent=2, ensure_ascii=False))

            # SHAP 图表
            for name, path in self.shap_figures.items():
                if os.path.exists(path):
                    zf.write(path, f'shap_{name}.png')
        buf.seek(0)
        zip_path = os.path.join(tempfile.gettempdir(), f"deep_predict_results_{uuid.uuid4().hex[:8]}.zip")
        with open(zip_path, 'wb') as f:
            f.write(buf.getvalue())
        return zip_path


# ============================================================
# ========== PredictVisualizer（Nature 配色）==============
# ============================================================

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

class PredictVisualizer:
    _DPI = 300
    _FONT_FAMILY = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']

    def __init__(self, figsize=(12, 5), palette=None):
        self.figsize = figsize
        self.palette = palette or NATURE_PALETTE

    def _new_fig(self):
        fig = Figure(figsize=self.figsize, dpi=self._DPI)
        FigureCanvasAgg(fig)
        return fig

    def plot_prediction(self, y_true, y_pred, train_size=0, model_name="Model", y_col="Value"):
        """时序预测对比图"""
        fig = self._new_fig()
        ax = fig.add_subplot(111)
        n_train = int(len(y_true) * (1 - train_size)) if train_size else 0
        x_all = range(len(y_true))
        x_train = x_all[:n_train] if n_train else []
        x_test = x_all[n_train:] if n_train else x_all

        # 全量数据
        ax.plot(x_all, y_true, color=self.palette[0], linewidth=1.5, label='真实值', alpha=0.8)
        ax.plot(x_all, y_pred, color=self.palette[1], linewidth=1.5, label='预测值', alpha=0.8)
        if n_train:
            ax.axvline(n_train - 1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='训练/测试分界')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(y_col)
        ax.set_title(f'{model_name} 预测结果', fontfamily='sans-serif')
        ax.legend(framealpha=0.3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_scatter(self, y_true, y_pred, model_name="Model"):
        """预测 vs 真实 散点图"""
        fig = self._new_fig()
        ax = fig.add_subplot(111)
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, c=self.palette[0])
        vmin = min(y_true.min(), y_pred.min())
        vmax = max(y_true.max(), y_pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1, label='y=x')
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(f'{model_name} 预测 vs 真实', fontfamily='sans-serif')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


# ============================================================
# ========== Gradio UI ==============================
# ============================================================

CSS = """
:root { --color-primary: #4DBBD5; }
.gradio-container { max-width: 1400px !important; margin: auto; }
.panel { background: #f8fafc; border-radius: 12px; padding: 1em; margin: 0.5em 0; }
h2 { color: #3C5488; }
.metric-label { color: #94a3b8; font-size: 0.85em; }
.metric-value { color: #22c55e; font-size: 1.5em; font-weight: 700; }
"""

# 全局状态
_dp_loader = DPDataLoader()
_dp_predictor = DPPredictor()
_dp_x_cols = []
_dp_y_col = ""


def on_load(file_obj):
    global _dp_loader, _dp_x_cols
    if file_obj is None:
        return "❌ 请先上传 CSV 文件", gr.update(choices=[])
    path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    try:
        n = _dp_loader.load(path)
        cols = list(_dp_loader.df.columns)
        info = _dp_loader.get_info()
        _dp_x_cols = [_ for _ in cols]
        return info, gr.update(choices=cols, value=cols[:1] if cols else None)
    except Exception as e:
        return f"❌ 加载失败: {e}", gr.update(choices=[], value=None)


def on_y_change(y_col):
    global _dp_y_col
    _dp_y_col = y_col or ""
    return ""


def on_train(model_name, test_size,
             lstm_epochs, lstm_hidden, lstm_seq, lstm_lr, lstm_bs,
             rf_trees, rf_depth,
             gb_trees, gb_depth, gb_lr,
             x_cols):
    global _dp_loader, _dp_predictor, _dp_x_cols, _dp_y_col

    if _dp_loader.df is None:
        return "❌ 请先加载数据", "", gr.update()
    if not x_cols:
        return "❌ 请至少选择一个特征列", "", gr.update()
    if not _dp_y_col:
        return "❌ 请选择目标列", "", gr.update()

    try:
        x_features = _dp_loader.get_feature_matrix(exclude_cols=[_dp_y_col])
        y_series = _dp_loader.df[_dp_y_col]
        selected = [c for c in x_features.columns if c in x_cols]
        X_df = x_features[selected]
    except Exception as e:
        return f"❌ 特征选择失败: {e}", "", gr.update()

    # 构建参数
    params = {}
    if model_name == 'LSTM':
        params = dict(epochs=int(lstm_epochs), hidden_size=int(lstm_hidden),
                     seq_len=int(lstm_seq), learning_rate=float(lstm_lr), batch_size=int(lstm_bs))
    elif model_name == 'PatchTST':
        params = dict(epochs=int(lstm_epochs), d_model=int(lstm_hidden),
                      seq_len=int(lstm_seq), lr=float(lstm_lr))

    # 训练
    success, msg, extra = _dp_predictor.train(X_df, y_series, _dp_y_col, model_name, params, float(test_size))
    if not success:
        return msg, "", gr.update()

    # 绘制图表
    try:
        preds = _dp_predictor.predict(X_df)
        y_true = y_series.values
        min_len = min(len(y_true), len(preds))
        y_true = y_true[:min_len]
        preds = preds[:min_len]
        viz = PredictVisualizer(figsize=(12, 5))
        fig = viz.plot_prediction(np.array(y_true), np.array(preds),
                                  train_size=float(test_size), model_name=model_name, y_col=_dp_y_col)
        fig_dpi300 = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        fig_dpi300.print_figure(buf, format='png', dpi=300)
        buf.seek(0)
        plt.close(fig)
        import base64
        img_data = base64.b64encode(buf.getvalue()).decode()
        plot_html = f'<img src="data:image/png;base64,{img_data}" style="width:100%"/>'
    except Exception as e:
        plot_html = f"<p>⚠️ 图表生成失败: {e}</p>"

    # 指标
    metrics_html = ""
    if _dp_predictor.metrics:
        metrics_html = "<div style='display:flex;gap:2em;flex-wrap:wrap;margin:1em 0'>"
        for k, v in _dp_predictor.metrics.items():
            metrics_html += f"<div><div class='metric-label'>{k}</div><div class='metric-value'>{v}</div></div>"
        metrics_html += "</div>"

    return msg, metrics_html, plot_html


def on_shap():
    global _dp_loader, _dp_predictor, _dp_x_cols
    if not _dp_predictor.is_fitted:
        return "❌ 请先训练模型", gr.update()
    if _dp_predictor._is_lstm or _dp_predictor._is_patchtst:
        return "⚠️ SHAP 仅支持 sklearn 模型（LSTM/PatchTST 除外）", gr.update()
    try:
        X_df = _dp_loader.get_feature_matrix(exclude_cols=[_dp_y_col])
        selected = [c for c in X_df.columns if c in _dp_x_cols]
        X_df = X_df[selected]
        figs, msg = _dp_predictor.run_shap_analysis(X_df)
        shap_html = ""
        if figs:
            import base64
            for name, path in figs.items():
                with open(path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                shap_html += f"<p><b>SHAP {name.title()}</b></p><img src='data:image/png;base64,{img_data}' style='width:100%'/>"
        return msg + (shap_html or "<p>无图表</p>"), gr.update()
    except Exception as e:
        return f"❌ SHAP 失败: {e}", gr.update()


def on_download():
    global _dp_loader, _dp_predictor, _dp_x_cols, _dp_y_col
    if not _dp_predictor.is_fitted:
        return None
    if not hasattr(_dp_predictor, 'model') or _dp_predictor.model is None:
        if not _dp_predictor._is_lstm and not _dp_predictor._is_patchtst:
            return None
    try:
        X_df = _dp_loader.get_feature_matrix(exclude_cols=[_dp_y_col])
        selected = [c for c in X_df.columns if c in _dp_x_cols]
        X_df = X_df[selected]
        y_series = _dp_loader.df[_dp_y_col]
        preds = _dp_predictor.predict(X_df)
        if preds is None:
            return None
        # 自动运行 SHAP
        if not _dp_predictor.shap_figures and not _dp_predictor._is_lstm and not _dp_predictor._is_patchtst:
            _dp_predictor.run_shap_analysis(X_df)
        return _dp_predictor.download_package(X_df, y_series, preds)
    except Exception:
        return None


# ============================================================
# ========== 构建 UI ==============================
# ============================================================

def build_ui():
    with gr.Blocks(title="DeepPredict 时序预测工具", css=CSS) as app:
        gr.Markdown("""
        <div style="text-align:center; padding: 1.5em 0;">
            <h1 style="font-size: 2.2em; color: #3C5488; margin: 0;">📈 DeepPredict 时序预测工具</h1>
            <p style="color: #64748b; font-size: 1em; margin: 0.5em 0 0 0;">
                LSTM · CNN1D · PatchTST · RandomForest · GradientBoosting · Linear · Logistic
            </p>
            <p style="color: #94a3b8; font-size: 0.85em;">
                Nature 配色 · 300 DPI · 无需 LaTeX · 一键 ZIP 导出
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.File(label="📂 上传 CSV 文件", file_count="single", file_types=[".csv"])
            with gr.Column(scale=1):
                load_btn = gr.Button("🚀 加载数据", variant="primary")
                data_info = gr.Markdown("**数据信息**：请先上传 CSV 文件")

        with gr.Row():
            x_col_multiselect = gr.Dropdown(multiselect=True, label="📌 选择特征列（X）")
            y_col_select = gr.Dropdown(label="🎯 选择目标列（Y）")
            test_size = gr.Slider(0.05, 0.4, value=0.2, step=0.05, label="测试集比例")

        gr.Markdown("---")

        with gr.Row():
            model = gr.Radio(
                ['RandomForest', 'GradientBoosting', 'LinearRegression',
                 'LogisticRegression', 'LSTM', 'PatchTST'],
                label="🤖 模型选择", value='RandomForest')

        with gr.Tab("🔧 RandomForest"):
            with gr.Row():
                rf_trees = gr.Number(label="树的数量", value=100, precision=0)
                rf_depth = gr.Number(label="最大深度（0=不限）", value=0, precision=0)
        with gr.Tab("🔧 GradientBoosting"):
            with gr.Row():
                gb_trees = gr.Number(label="树的数量", value=100, precision=0)
                gb_depth = gr.Number(label="最大深度", value=3, precision=0)
                gb_lr = gr.Number(label="学习率", value=0.1)
        with gr.Tab("🔧 LSTM"):
            with gr.Row():
                lstm_epochs = gr.Number(label="训练轮次", value=20, precision=0)
                lstm_hidden = gr.Number(label="隐藏层大小", value=64, precision=0)
                lstm_seq = gr.Number(label="序列长度", value=30, precision=0)
                lstm_lr = gr.Number(label="学习率", value=0.001)
                lstm_bs = gr.Number(label="批次大小", value=32, precision=0)
        with gr.Tab("🔧 PatchTST"):
            with gr.Row():
                pt_epochs = gr.Number(label="训练轮次", value=20, precision=0)
                pt_hidden = gr.Number(label="模型维度", value=128, precision=0)
                pt_seq = gr.Number(label="序列长度", value=30, precision=0)
                pt_lr = gr.Number(label="学习率", value=0.001)
        with gr.Tab("🔧 Linear / Logistic"):
            gr.Markdown("线性模型无需额外参数")

        with gr.Row():
            train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
            shap_btn = gr.Button("🔍 SHAP 分析", size="lg")
            download_btn = gr.Button("📦 下载完整结果包（ZIP）", variant="secondary", size="lg")

        with gr.Row():
            train_msg = gr.Textbox(label="训练状态", lines=2, interactive=False)
            metrics_out = gr.HTML(label="📊 评估指标")
        result_plot = gr.HTML(label="📈 预测结果图")
        shap_out = gr.HTML(label="🔍 SHAP 分析结果")
        download_file = gr.File(label="💾 下载结果")

        # 事件绑定
        load_btn.click(on_load, inputs=[file_input], outputs=[data_info, x_col_multiselect])
        y_col_select.change(on_y_change, inputs=[y_col_select], outputs=[])
        train_btn.click(on_train,
                       inputs=[model, test_size,
                               lstm_epochs, lstm_hidden, lstm_seq, lstm_lr, lstm_bs,
                               rf_trees, rf_depth,
                               gb_trees, gb_depth, gb_lr,
                               x_col_multiselect],
                       outputs=[train_msg, metrics_out, result_plot])
        shap_btn.click(on_shap, outputs=[shap_out, shap_out])
        download_btn.click(on_download, outputs=[download_file])

        gr.Markdown("""
        <div style="text-align:center; padding: 1em; color: #64748b; font-size: 0.8em;">
            DeepPredict v2.0 · © 2026 · Nature 配色 · 300 DPI · Powered by PyTorch · scikit-learn · SHAP
        </div>
        """)

    return app


# ============================================================
# ========== 启动 ==============================
# ============================================================

def main():
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        theme=gr.themes.Glass(),
    )


if __name__ == "__main__":
    main()
