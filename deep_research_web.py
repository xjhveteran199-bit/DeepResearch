"""
Deep-Research 统一平台 Web 界面 v2.0
整合 DeepPredict / DeepClassify / DeepDetect 三大模块

升级日志 v2.0:
- [DeepPredict] 多模态输入：支持选择多个 X 特征列 + 1 个 Y 目标列
- [DeepPredict] SHAP 可解释性分析（参考 adi6492 论文方法）
- [新增] 统一入口，三模块Tab切换
"""

import os
import time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "7860"

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

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
import warnings
warnings.filterwarnings('ignore')

# ===== DeepPredict 路径 =====
DEEP_PREDICT_PATH = Path(r"C:\Users\XJH\DeepResearch\DeepPredict")
DEEP_DETECT_PATH = Path(r"C:\Users\XJH\DeepResearch\DeepDetect")
DEEP_CLASSIFY_PATH = Path(r"C:\Users\XJH\DeepResearch\DeepClassify")
sys.path.insert(0, str(DEEP_PREDICT_PATH))

# ===== DeepPredict Visualizer =====
sys.path.insert(0, str(DEEP_PREDICT_PATH / "src"))
from visualizer import PredictVisualizer

# ===== DeepPredict 高级模型（提前导入，避免子进程路径丢失）=====
sys.path.insert(0, str(DEEP_PREDICT_PATH / "src"))
try:
    from models.lstm_model import LSTMPredictor
except ImportError:
    LSTMPredictor = None
try:
    from models.patchtst_model import PatchTSTPredictor
except ImportError:
    PatchTSTPredictor = None

# ===== 配置日志 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("DeepResearch")

# ===== 统一配色主题 =====
CSS = """
:root {
    --color-primary: #6366f1;
    --color-secondary: #8b5cf6;
    --color-accent: #06b6d4;
    --color-bg: #0f172a;
    --color-surface: #1e293b;
    --color-text: #f8fafc;
    --color-muted: #94a3b8;
    --color-success: #22c55e;
    --color-warning: #f59e0b;
    --color-danger: #ef4444;
}
body { background: var(--color-bg); color: var(--color-text); font-family: 'Segoe UI', sans-serif; }
.gradio-container { max-width: 1400px !important; }
.module-header { font-size: 1.4em; font-weight: 700; color: #818cf8; padding: 0.5em 0; }
.module-desc { color: #94a3b8; font-size: 0.95em; margin-bottom: 1em; }
.metric-card { background: #1e293b; border-radius: 12px; padding: 1em; border: 1px solid #334155; }
.metric-label { color: #94a3b8; font-size: 0.85em; }
.metric-value { color: #22c55e; font-size: 1.5em; font-weight: 700; }
"""

# ============================================================
# ========== DeepPredict 模块（已完整功能）===============
# ============================================================

class DPDataLoader:
    """DeepPredict 专用数据加载器"""
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

        self._analyze()
        return True, f"加载成功：{self.df.shape[0]}行 × {self.df.shape[1]}列"

    def _analyze(self):
        self.numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        self.categorical_cols = list(self.df.select_dtypes(include=['object', 'category']).columns)
        self.all_cols = list(self.df.columns)

    def get_preview(self, n=20):
        if self.df is None:
            return pd.DataFrame()
        return self.df.head(n)

    def get_info(self):
        if self.df is None:
            return ""
        info = f"**数据形状**：{self.df.shape[0]} 行 × {self.df.shape[1]} 列\n\n"
        info += f"**数值列**（{len(self.numeric_cols)}）：`{'`, `'.join(self.numeric_cols[:8])}{'...' if len(self.numeric_cols)>8 else ''}`\n\n"
        info += f"**类别列**（{len(self.categorical_cols)}）：`{'`, `'.join(self.categorical_cols[:5])}{'...' if len(self.categorical_cols)>5 else ''}`"
        return info


class DPPredictor:
    """DeepPredict 预测器（增强版：支持多X列 + SHAP）"""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.task_type = None
        self.feature_names = []
        self.target_col = None
        self.is_fitted = False
        self.metrics = {}
        self._lstm_model = None
        self._is_lstm = False
        self._is_patchtst = False
        self.shap_analyzer = None
        self.shap_figures = {}

    def train(self, X_df, y_series, target_col, model_name, params, test_size=0.2):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        try:
            self.target_col = target_col
            self.feature_names = list(X_df.columns)
            # P2修复：只对数值列做 fillna，避免 Date/字符串列导致 TypeError
            numeric_mask = X_df.select_dtypes(include=[np.number]).columns
            X = X_df[numeric_mask].fillna(X_df[numeric_mask].median())
            if X.shape[1] == 0:
                return False, "❌ 选中特征中没有任何数值列，无法进行回归/分类"

            # 判断任务类型
            if model_name in ['LSTM', 'PatchTST']:
                self.task_type = 'time_series_regression'
            elif y_series.dtype in ['int64'] and y_series.nunique() <= 10:
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'

            # ===== LSTM =====
            if model_name == 'LSTM':
                self._is_lstm = True
                X_arr = X.values.astype(np.float32)
                y_arr = y_series.values.astype(np.float32)
                if LSTMPredictor is None:
                    return False, "❌ LSTMPredictor 未正确加载，请检查安装", {}
                self._lstm_model = LSTMPredictor()
                self._lstm_model = LSTMPredictor()
                p = params or {}
                success, msg = self._lstm_model.train(
                    X_arr, y_arr,
                    hidden_size=p.get('hidden_size', 64),
                    num_layers=p.get('num_layers', 3),
                    epochs=p.get('epochs', 80),
                    batch_size=p.get('batch_size', 32),
                    learning_rate=p.get('learning_rate', 0.001),
                    seq_len=p.get('seq_len', 90),
                    test_size=test_size,
                    target_col=target_col
                )
                self.metrics = self._lstm_model.metrics
                self.is_fitted = True
                return success, msg

            # ===== PatchTST =====
            if model_name == 'PatchTST':
                self._is_patchtst = True
                X_arr = X.values.astype(np.float32)
                y_arr = y_series.values.astype(np.float32)
                if PatchTSTPredictor is None:
                    return False, "❌ PatchTST 未正确加载，请检查安装（pip install prophet 或检查依赖）", {}
                self._lstm_model = PatchTSTPredictor()
                self._lstm_model = PatchTSTPredictor()
                p = params or {}
                success, msg = self._lstm_model.train(
                    X_arr, y_arr,
                    seq_len=p.get('seq_len', 96),
                    pred_len=p.get('pred_len', 96),
                    patch_size=p.get('patch_size', 16),
                    d_model=p.get('d_model', 128),
                    n_heads=p.get('n_heads', 4),
                    n_layers=p.get('n_layers', 3),
                    d_ff=p.get('d_ff', 256),
                    epochs=p.get('epochs', 30),
                    batch_size=p.get('batch_size', 32),
                    learning_rate=p.get('learning_rate', 0.0005),
                    test_size=test_size,
                    target_col=target_col
                )
                self.metrics = self._lstm_model.metrics
                self.is_fitted = True
                return success, msg

            # ===== sklearn 模型 =====
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            if self.task_type == 'classification':
                self.label_encoder = LabelEncoder()
                y_enc = self.label_encoder.fit_transform(y_series.astype(str))
            else:
                y_enc = y_series.astype(float)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_enc, test_size=test_size, random_state=42
            )

            # P0修复：LogisticRegression 是分类器，不能用于回归，明确报错而非静默替换
            if model_name == 'LogisticRegression' and self.task_type == 'regression':
                return False, "❌ LogisticRegression 是分类器，不能用于回归任务。请选择 LinearRegression、Ridge 或 ElasticNet。", {}

            model_map = {
                ('RandomForest', 'regression'): RandomForestRegressor,
                ('RandomForest', 'classification'): RandomForestClassifier,
                ('GradientBoosting', 'regression'): GradientBoostingRegressor,
                ('GradientBoosting', 'classification'): GradientBoostingClassifier,
                ('LinearRegression', 'regression'): LinearRegression,
                ('LogisticRegression', 'classification'): LogisticRegression,
            }
            key = (model_name, self.task_type)
            model_cls = model_map.get(key)
            self.model = model_cls(**(params or {}))
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            if self.task_type == 'classification':
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                self.metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
                msg = f"✅ {model_name} 训练完成！\n准确率={acc:.2%} 精确率={prec:.2%} 召回率={rec:.2%} F1={f1:.4f}"
            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                self.metrics = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
                msg = f"✅ {model_name} 训练完成！\nR²={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f}"

            self.is_fitted = True
            return True, msg

        except Exception as e:
            logger.error(f"DPPredictor.train 错误: {e}")
            return False, f"❌ 训练失败: {str(e)}"

    def run_shap_analysis(self, X_df):
        """运行 SHAP 分析（仅 sklearn 模型）"""
        if not self.is_fitted or self._is_lstm or self._is_patchtst or self.model is None:
            return None, "⚠️ 当前模型不支持 SHAP（仅支持 sklearn 模型）"

        try:
            sys.path.insert(0, str(DEEP_PREDICT_PATH / "src" / "utils"))
            from shap_analyzer import SHAPAnalyzer

            self.shap_analyzer = SHAPAnalyzer()
            X = X_df.fillna(X_df.median())
            success = self.shap_analyzer.fit(self.model, X, method='tree')

            if not success:
                return None, "⚠️ SHAP 分析失败，请检查模型类型"

            # 生成图表并保存为临时文件（Gradio 6.x 需要路径而非 BytesIO）
            import tempfile, os, uuid
            figs = {}
            fig_imp = self.shap_analyzer.plot_importance()
            if fig_imp:
                imp_path = os.path.join(tempfile.gettempdir(), f"shap_importance_{uuid.uuid4().hex[:8]}.png")
                fig_imp.savefig(imp_path, format='png', dpi=120, bbox_inches='tight')
                figs['importance'] = imp_path

            fig_beeswarm = self.shap_analyzer.plot_beeswarm()
            if fig_beeswarm:
                bsw_path = os.path.join(tempfile.gettempdir(), f"shap_beeswarm_{uuid.uuid4().hex[:8]}.png")
                fig_beeswarm.savefig(bsw_path, format='png', dpi=120, bbox_inches='tight')
                figs['beeswarm'] = bsw_path

            self.shap_figures = figs
            report = self.shap_analyzer.generate_report()
            return figs, report

        except Exception as e:
            logger.error(f"SHAP 分析错误: {e}")
            return None, f"⚠️ SHAP 分析出错: {str(e)}"

    def predict(self, X_df):
        if not self.is_fitted:
            raise ValueError("模型未训练")
        if self.model is None and not self._is_lstm and not self._is_patchtst:
            raise ValueError("模型未正确训练，无法预测")
        # P2修复：只对数值列做 fillna
        numeric_mask = X_df.select_dtypes(include=[np.number]).columns
        X = X_df[numeric_mask].fillna(X_df[numeric_mask].median())
        if self._is_lstm or self._is_patchtst:
            return self._lstm_model.predict(X.values.astype(np.float32))
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        if self.task_type == 'classification' and self.label_encoder:
            pred = self.label_encoder.inverse_transform(pred.astype(int))
        return pred

    def download_package(self, X_df, y_df, predictions):
        """生成结果 zip 包并返回临时文件路径（Gradio 6.x 兼容）"""
        import tempfile, os, uuid
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 预测结果 CSV
            result_df = X_df.copy()
            # 长度对齐（处理 LSTM/PatchTST 输出长度与原始数据不一致的问题）
            min_len = min(len(result_df), len(y_df.values), len(predictions))
            if min_len < len(result_df):
                result_df = result_df.iloc[:min_len].reset_index(drop=True)
            y_vals = np.asarray(y_df.values)[:min_len]
            pred_vals = np.asarray(predictions)[:min_len]
            result_df[self.target_col] = y_vals
            result_df['prediction'] = pred_vals
            zf.writestr('forecast_data.csv', result_df.to_csv(index=False))
            # 指标 JSON
            zf.writestr('metrics.json', json.dumps(self.metrics, indent=2))
            # SHAP 图（如果有）— run_shap_analysis 已存为路径
            if self.shap_figures:
                for fig_name, fig_path in self.shap_figures.items():
                    if os.path.isfile(fig_path):
                        zf.writestr(f'shap_{fig_name}.png', open(fig_path, 'rb').read())
        buf.seek(0)
        # 保存为临时文件，返回路径
        zip_path = os.path.join(tempfile.gettempdir(), f"deep_predict_results_{uuid.uuid4().hex[:8]}.zip")
        with open(zip_path, 'wb') as f:
            f.write(buf.getvalue())
        return zip_path


# ===== DeepPredict Gradio UI Builder =====
def build_deep_predict_ui():
    with gr.Tab("📈 DeepPredict"):
        gr.Markdown("## DeepPredict — 时序 / 回归预测")
        gr.Markdown("上传 CSV，选择特征列（X）和目标列（Y），训练模型并预测未来趋势。支持多模态输入 + SHAP 可解释性分析。")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ 上传数据")
                file_input = gr.File(label="上传 CSV 文件", file_types=[".csv"])
                load_btn = gr.Button("加载数据", variant="primary")

                gr.Markdown("### 2️⃣ 数据预览")
                data_info = gr.Markdown("请先上传数据...")
                preview_table = gr.DataFrame(label="数据预览（前20行）", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 3️⃣ 特征与目标列选择")
                gr.Markdown("**X 特征列**（多选，支持多模态输入）：选择作为输入特征的列")
                x_cols = gr.CheckboxGroup(label="X 特征列（可多选）", choices=[])
                gr.Markdown("**Y 目标列**（单选）：要预测的目标变量")
                y_col = gr.Dropdown(label="Y 目标列", choices=[], value=None)

                gr.Markdown("### 4️⃣ 模型选择")
                model_choice = gr.Dropdown(
                    label="选择模型",
                    choices=['RandomForest', 'GradientBoosting', 'LinearRegression',
                             'LogisticRegression', 'LSTM', 'PatchTST'],
                    value='RandomForest'
                )
                test_size = gr.Slider(label="测试集比例", minimum=0.1, maximum=0.4, value=0.2, step=0.05)

                train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")

        with gr.Row():
            metrics_output = gr.JSON(label="📊 训练指标")
            result_plot = gr.Plot(label="📉 预测结果图")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 5️⃣ SHAP 可解释性分析")
                gr.Markdown("分析每个输入特征对预测结果的贡献度（参考 adi6492 论文 SHAP 方法）")
                shap_btn = gr.Button("🔍 运行 SHAP 分析", variant="secondary")
                shap_output = gr.Markdown("")
                shap_plot_importance = gr.Image(label="特征重要性图")
                shap_plot_beeswarm = gr.Image(label="SHAP 分布图")

            with gr.Column():
                gr.Markdown("### 6️⃣ 下载结果")
                download_btn = gr.Button("📦 下载完整结果包（CSV + 指标 + SHAP图）", variant="secondary")
                download_file = gr.File(label="下载")
                predict_btn = gr.Button("🔮 预测新数据", variant="secondary")

        # ===== 状态 =====
        # ===== 全局状态（Gradio 6.x 兼容性：避免 gr.State 的 subscript 问题）=====
        global _dp_loader, _dp_predictor, _dp_x_cols, _dp_y_col
        _dp_loader = DPDataLoader()
        _dp_predictor = DPPredictor()
        _dp_x_cols = []
        _dp_y_col = None

        # ===== 事件绑定（直接操作全局变量，无 gr.State）=====
        def on_load(file_obj):
            global _dp_loader, _dp_x_cols
            if file_obj is None:
                return "请上传 CSV 文件", gr.update(choices=[]), gr.update(choices=[]), gr.update()
            path = _extract_path(file_obj)
            if not path:
                return "❌ 无法读取文件", gr.update(choices=[]), gr.update(choices=[]), gr.update()
            success, msg = _dp_loader.load(path)
            if not success:
                return msg, gr.update(choices=[]), gr.update(choices=[]), gr.update()
            _dp_x_cols = []
            choices = _dp_loader.all_cols
            preview = _dp_loader.get_preview()
            info = _dp_loader.get_info()
            return info, gr.update(choices=choices), gr.update(choices=choices, value=None), preview

        load_btn.click(on_load, inputs=[file_input],
                       outputs=[data_info, x_cols, y_col, preview_table])

        def on_x_selected(choices):
            global _dp_x_cols
            _dp_x_cols = list(choices) if choices else []
            return

        x_cols.change(on_x_selected, inputs=[x_cols], outputs=[])

        def on_y_selected(value):
            global _dp_y_col
            _dp_y_col = value
            return

        y_col.change(on_y_selected, inputs=[y_col], outputs=[])

        def on_train(model_name, test_size):
            global _dp_loader, _dp_predictor, _dp_x_cols, _dp_y_col

            if not _dp_x_cols or not _dp_y_col:
                return "❌ 请先选择 X 特征列和 Y 目标列", {}, gr.update()
            if _dp_loader.df is None:
                return "❌ 请先加载数据", {}, gr.update()

            X_df = _dp_loader.df[_dp_x_cols]
            y_series = _dp_loader.df[_dp_y_col]

            params = {}
            if model_name == 'LSTM':
                n = len(_dp_loader.df)
                params['seq_len'] = min(90, max(30, n // 5))
                params['hidden_size'] = 64
                params['num_layers'] = 3
                params['epochs'] = 80
            elif model_name == 'PatchTST':
                n = len(_dp_loader.df)
                params['seq_len'] = min(96, max(12, n // 5))
                params['pred_len'] = params['seq_len'] // 2
                params['epochs'] = 20

            success, msg = _dp_predictor.train(X_df, y_series, _dp_y_col, model_name, params, float(test_size))
            if not success:
                return msg, {}, gr.update()
            fig = _plot_results(_dp_predictor, X_df, y_series, _dp_y_col)
            return msg, _dp_predictor.metrics, fig

        train_btn.click(on_train, inputs=[model_choice, test_size],
                        outputs=[data_info, metrics_output, result_plot])

        def on_shap():
            global _dp_loader, _dp_predictor, _dp_x_cols
            if not _dp_predictor.is_fitted:
                return "⚠️ 请先训练模型", None, None
            X_df = _dp_loader.df[_dp_x_cols]
            figs, report = _dp_predictor.run_shap_analysis(X_df)
            if figs is None:
                return report, None, None
            return report, figs.get('importance'), figs.get('beeswarm')

        shap_btn.click(on_shap, inputs=[],
                       outputs=[shap_output, shap_plot_importance, shap_plot_beeswarm])

        def on_download():
            global _dp_loader, _dp_predictor, _dp_x_cols, _dp_y_col
            if _dp_predictor is None or not _dp_predictor.is_fitted:
                return None
            # 防御：确保 model 已正确初始化
            if not hasattr(_dp_predictor, 'model') or _dp_predictor.model is None:
                return None
            X_df = _dp_loader.df[_dp_x_cols]
            y_series = _dp_loader.df[_dp_y_col]
            preds = _dp_predictor.predict(X_df)
            if preds is None:
                return None
            return _dp_predictor.download_package(X_df, y_series, preds)

        download_btn.click(on_download, inputs=[], outputs=[download_file])


# ============================================================
# ========== DeepClassify 模块（子Agent构建中）===============
# ============================================================

# ============================================================
# ========== DeepClassify 模块（已集成）=====================
# ============================================================

def _build_deep_classify_ui():
    """DeepClassify Tab — 集成 DeepClassify 模块"""
    import sys as _sys
    _sys.path.insert(0, str(DEEP_CLASSIFY_PATH))
    _sys.path.insert(0, str(DEEP_CLASSIFY_PATH / "src"))

    from core.data_loader import DataLoader as DCLDataLoader
    from core.metrics import ClassificationMetrics
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt

    # 延迟导入分类器，避免 torch 导入失败时影响其他分类器
    HAS_TORCH = True
    _torch_err = ""

    # ===== 状态 =====
    dc_state = gr.State({
        'loader': None, 'classifier': None, 'X': None,
        'y': None, 'y_enc': None, 'le': None,
        'class_names': [], 'metrics': {}, 'predict_df': None
    })

    gr.Markdown("## 🏷️ DeepClassify — 信号 / 数据分类")
    gr.Markdown("上传带标签 CSV → 选特征/标签列 → 选模型（CNN1D/RF/GB/SVM）→ 训练评估 → 预测新数据")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 数据加载")
            dc_file = gr.File(label="上传 CSV（含特征+标签列）", file_types=[".csv"])
            dc_load_btn = gr.Button("加载", variant="primary")
            dc_info = gr.Markdown("请上传 CSV...")
            dc_preview = gr.DataFrame(label="预览", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 🏷️ 标签 & 特征列选择")
            dc_target = gr.Dropdown(label="标签列（目标分类列）", choices=[], value=None)
            dc_test_size = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="测试集比例")
            dc_select_btn = gr.Button("确认列选择", variant="secondary")
            dc_col_info = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 选择模型")
            dc_model = gr.Dropdown(
                label="分类模型",
                choices=['CNN1D', 'RandomForest', 'GradientBoosting', 'SVM'],
                value='RandomForest'
            )
            # CNN1D 参数
            with gr.Column(visible=False) as dc_cnn_panel:
                gr.Markdown("**CNN1D 参数**")
                dc_cnn_hidden = gr.Number(label="hidden_channels", value=128)
                dc_cnn_kernel = gr.Number(label="kernel_size", value=5)
                dc_cnn_epochs = gr.Number(label="epochs", value=50)
                dc_cnn_lr = gr.Number(label="learning_rate", value=0.001)
                dc_cnn_bs = gr.Number(label="batch_size", value=32)
            # RF 参数
            with gr.Column(visible=False) as dc_rf_panel:
                gr.Markdown("**RandomForest 参数**")
                dc_rf_trees = gr.Number(label="n_estimators", value=100)
                dc_rf_depth = gr.Number(label="max_depth (0=无限制)", value=0)
            # GB 参数
            with gr.Column(visible=False) as dc_gb_panel:
                gr.Markdown("**GradientBoosting 参数**")
                dc_gb_trees = gr.Number(label="n_estimators", value=100)
                dc_gb_depth = gr.Number(label="max_depth", value=3)
                dc_gb_lr = gr.Number(label="learning_rate", value=0.1)
            # SVM 参数
            with gr.Column(visible=False) as dc_svm_panel:
                gr.Markdown("**SVM 参数**")
                dc_svm_c = gr.Number(label="C", value=1.0)
                dc_svm_kernel = gr.Dropdown(label="kernel", choices=['rbf', 'linear', 'poly'], value='rbf')

            dc_train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")

    with gr.Row():
        dc_train_msg = gr.Textbox(label="训练状态", lines=2, interactive=False)
        dc_metrics_out = gr.HTML(label="📊 评估指标")
        dc_cm_out = gr.HTML(label="📋 混淆矩阵")
        dc_roc_out = gr.Plot(label="📈 ROC 曲线")
        dc_model_file = gr.File(label="💾 下载模型")

    with gr.Row():
        dc_pred_file = gr.File(label="📄 上传待预测CSV", file_types=[".csv"])
        dc_pred_btn = gr.Button("🔮 执行预测", variant="secondary")
        dc_pred_out = gr.HTML(label="预测结果")
        dc_pred_download = gr.File(label="下载预测结果")

    # ===== 事件绑定 =====
    def dc_on_load(file_obj):
        if file_obj is None:
            return gr.update(), "请上传文件", gr.DataFrame(), dc_state
        path = _extract_path(file_obj)
        if not path:
            return gr.update(), "无法读取", gr.DataFrame(), dc_state
        try:
            loader = DCLDataLoader()
            ok, msg = loader.load_csv(path)
            if not ok:
                return gr.update(), f"❌ {msg}", gr.DataFrame(), dc_state
            new_state = {'loader': loader, 'classifier': None, 'X': None,
                         'y': None, 'y_enc': None, 'le': None,
                         'class_names': [], 'metrics': {}, 'predict_df': None}
            return (gr.update(choices=list(loader.df.columns), value=None),
                    f"✅ {msg}\n数值列: {len(loader.get_summary()['numeric_cols'])}\n类别列: {len(loader.get_summary()['categorical_cols'])}",
                    loader.get_preview(20),
                    new_state)
        except Exception as e:
            return gr.update(), f"❌ {e}", gr.DataFrame(), dc_state

    dc_load_btn.click(dc_on_load, inputs=[dc_file],
                      outputs=[dc_target, dc_info, dc_preview, dc_state])

    def dc_on_select(target_col, test_size, state):
        loader = state['loader']
        if loader is None or target_col is None:
            return "", dc_state
        try:
            loader.select_target(target_col)
            X = loader.get_feature_matrix(exclude_cols=[target_col])
            # y_enc = LabelEncoder 编码值（用于训练），y = 原始字符串标签（用于 metric 计算）
            y_enc, le = loader.get_target_encoded()
            y = loader.df[target_col].astype(str)  # 原始字符串标签
            class_names = list(le.classes_)
            info = (f"✅ 目标列: {target_col}\n"
                    f"类别: {class_names}\n"
                    f"特征数: {X.shape[1]} | 样本数: {len(X)}\n"
                    f"训练/测试: {(1-float(test_size)):.0%}/{float(test_size):.0%}")
            new_state = dict(state)
            new_state.update({'X': X, 'y': y, 'y_enc': y_enc, 'le': le,
                             'class_names': class_names})
            return info, new_state
        except Exception as e:
            return f"❌ {e}", state

    dc_select_btn.click(dc_on_select, inputs=[dc_target, dc_test_size, dc_state],
                       outputs=[dc_col_info, dc_state])

    def dc_on_model_change(model_name):
        vis_cnn = model_name == 'CNN1D'
        vis_rf = model_name == 'RandomForest'
        vis_gb = model_name == 'GradientBoosting'
        vis_svm = model_name == 'SVM'
        return (gr.update(visible=vis_cnn), gr.update(visible=vis_rf),
                gr.update(visible=vis_gb), gr.update(visible=vis_svm))

    dc_model.change(dc_on_model_change, inputs=[dc_model],
                    outputs=[dc_cnn_panel, dc_rf_panel, dc_gb_panel, dc_svm_panel])

    def dc_on_train(model_name, test_size,
                    cnn_hidden, cnn_kernel, cnn_epochs, cnn_lr, cnn_bs,
                    rf_trees, rf_depth,
                    gb_trees, gb_depth, gb_lr,
                    svm_c, svm_kernel,
                    state):
        loader = state['loader']
        if loader is None or state['X'] is None:
            return "❌ 请先加载数据并选择列", "", "", None, None, state
        try:
            X = state['X']
            y_enc = state['y_enc']
            le = state['le']
            class_names = state['class_names']

            # 构建分类器（直接导入，避免 sys.path 中 DeepPredict 遮蔽 DeepClassify 的 models）
            import sys as _sys
            _sys.path.insert(0, str(DEEP_CLASSIFY_PATH / "src"))
            if model_name == 'CNN1D':
                try:
                    from models import CNN1DClassifyWrapper
                except ImportError as _e:
                    return f"❌ CNN1D 导入失败: {_e}", "", "", None, None, state
                clf = CNN1DClassifyWrapper(
                    hidden_channels=int(cnn_hidden), kernel_size=int(cnn_kernel),
                    epochs=int(cnn_epochs), learning_rate=float(cnn_lr),
                    batch_size=int(cnn_bs))
            elif model_name == 'RandomForest':
                from models import RFClassifier
                clf = RFClassifier(
                    n_estimators=int(rf_trees),
                    max_depth=int(rf_depth) if int(rf_depth) > 0 else None)
            elif model_name == 'GradientBoosting':
                from models import GBClassifier
                clf = GBClassifier(
                    n_estimators=int(gb_trees), max_depth=int(gb_depth),
                    learning_rate=float(gb_lr))
            else:
                from models import SVMClassifier
                clf = SVMClassifier(C=float(svm_c), kernel=str(svm_kernel))

            ok, msg = clf.fit(X, y_enc)
            if not ok:
                return f"❌ {msg}", "", "", None, None, state

            # 评估
            # 使用原始标签（string）做 train/test split，确保与分类器返回的 string 标签类型一致
            X_arr = X.values.astype(np.float32)
            y_str = y.values  # 原始 string 标签
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_arr, y_str, test_size=float(test_size), random_state=42)
            y_pred = clf.predict(X_te)
            y_proba = clf.predict_proba(X_te)

            mc = ClassificationMetrics()
            # y_te 和 y_pred 都是原始 string 标签（分类器 inverse_transform 返回值）
            # 注意：RF/GB/SVM 也返回 string 标签，所以 metric 计算直接用 string
            full_metrics = mc.compute(
                y_true=np.array(y_te), y_pred=np.array(y_pred),
                y_proba=y_proba, labels=class_names)

            # 混淆矩阵（y_te 和 y_pred 都是原始标签，直接用 class_names 作为 labels）
            cm = confusion_matrix(y_te, y_pred, labels=class_names)
            cm_rows = []
            cm_rows.append("<tr><th></th>" + "".join([f"<th>{l}</th>" for l in class_names]) + "</tr>")
            for i, row in enumerate(cm):
                cm_rows.append("<tr>" + f"<td><b>{class_names[i]}</b></td>"
                            + "".join([f"<td>{v}</td>" for v in row]) + "</tr>")
            cm_html = f"<table style='border-collapse:collapse;width:100%'><caption><b>混淆矩阵</b></caption>"
            cm_html += "".join(cm_rows) + "</table>"

            # 指标 HTML
            m_rows = []
            for k, v in full_metrics.items():
                if isinstance(v, float):
                    m_rows.append(f"<tr><td><b>{k}</b></td><td>{v:.4f}</td></tr>")
                else:
                    m_rows.append(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>")
            metrics_html = f"<table style='border-collapse:collapse;width:100%'><caption><b>评估指标</b></caption>"
            metrics_html += "".join(m_rows) + "</table>"

            # ROC 曲线（y_te 现在是原始 string 标签）
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            if y_proba.shape[1] == 2:
                # 二分类：用 class_names[1] 作为正类
                fpr, tpr, _ = roc_curve((y_te == class_names[1]).astype(int), y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
            else:
                # 多分类：用 class_names[i] 作为第 i 类的二元标签
                for i in range(min(y_proba.shape[1], 5)):
                    fpr, tpr, _ = roc_curve((y_te == class_names[i]).astype(int), y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax_roc.plot(fpr, tpr, lw=2, label=f'Class {class_names[i]} (AUC={roc_auc:.3f})')
            ax_roc.plot([0,1],[0,1],'k--',lw=1)
            ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
            ax_roc.set_title('ROC Curve'); ax_roc.legend(); ax_roc.grid(alpha=0.3)
            plt.tight_layout()

            # 保存模型
            save_path = str(DEEP_CLASSIFY_PATH / "deepclassify_model.pkl")
            clf.save(save_path)

            new_state = dict(state)
            new_state.update({'classifier': clf, 'metrics': full_metrics})
            return msg, metrics_html, cm_html, fig_roc, save_path, new_state
        except Exception as e:
            import traceback
            return f"❌ 训练失败: {e}\n```\n{traceback.format_exc()}\n```", "", "", None, None, state

    dc_train_btn.click(dc_on_train,
                        inputs=[dc_model, dc_test_size,
                                dc_cnn_hidden, dc_cnn_kernel, dc_cnn_epochs, dc_cnn_lr, dc_cnn_bs,
                                dc_rf_trees, dc_rf_depth,
                                dc_gb_trees, dc_gb_depth, dc_gb_lr,
                                dc_svm_c, dc_svm_kernel,
                                dc_state],
                        outputs=[dc_train_msg, dc_metrics_out, dc_cm_out, dc_roc_out, dc_model_file, dc_state])

    def dc_on_pred(file_obj, state):
        if file_obj is None:
            return "⚠️ 请上传文件", gr.update()
        path = _extract_path(file_obj)
        if not path:
            return "❌ 无法读取文件", gr.update()
        clf = state.get('classifier')
        X = state.get('X')
        if clf is None or X is None:
            return "❌ 请先训练模型", gr.update()
        try:
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except:
                    continue
            if df is None:
                df = pd.read_csv(path, encoding='utf-8', errors='replace')
            # 对齐特征
            n_feat = X.shape[1]
            if df.shape[1] >= n_feat:
                X_pred = df.iloc[:, :n_feat].values.astype(np.float32)
            else:
                X_pred = df.values.astype(np.float32)
            y_pred = clf.predict(X_pred)
            y_proba = clf.predict_proba(X_pred)
            result_df = pd.DataFrame({'预测类别': y_pred, '预测概率': y_proba.max(axis=1).round(4)})
            result_html = result_df.head(50).to_string()
            save_path = str(DEEP_CLASSIFY_PATH / "predictions.csv")
            result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            new_state = dict(state)
            new_state['predict_df'] = df
            return f"✅ 预测完成！共 {len(y_pred)} 条\n\n{result_html}", gr.update(value=save_path)
        except Exception as e:
            return f"❌ 预测失败: {e}", gr.update()

    dc_pred_btn.click(dc_on_pred, inputs=[dc_pred_file, dc_state],
                      outputs=[dc_pred_out, dc_pred_download])

    return dc_state


def build_deep_classify_ui():
    with gr.Tab("🏷️ DeepClassify"):
        try:
            _build_deep_classify_ui()
        except Exception as e:
            gr.Markdown(f"❌ DeepClassify 加载失败: {e}")
            gr.Markdown("请确认依赖已安装：`pip install -r DeepClassify/requirements.txt`")


# ============================================================
# ========== DeepDetect 模块（已集成）====================
# ============================================================

def _build_deep_detect_ui():
    """DeepDetect Tab — 直接复用独立 app 的逻辑"""
    sys.path.insert(0, str(DEEP_DETECT_PATH))

    from DeepDetect.src.core.data_loader import DataLoader as DDLoader
    from DeepDetect.src.core.eval import evaluate_detector, format_metrics_table
    from DeepDetect.src.models.isolation_forest import IsolationForestDetector
    from DeepDetect.src.models.autoencoder import AutoencoderDetector
    from DeepDetect.src.models.ocsvm import OCSVMDetector
    from DeepDetect.src.models.lof_detector import LOFDetector
    from DeepDetect.src.models.stats_detector import StatsDetector
    from DeepDetect.src.models.lstm_detector import LSTMDetector

    DETECTOR_MAP = {
        'IsolationForest': IsolationForestDetector,
        'Autoencoder': AutoencoderDetector,
        'OneClassSVM': OCSVMDetector,
        'LOF': LOFDetector,
        'LSTM': LSTMDetector,
        'Stats_ZScore': lambda **kw: StatsDetector(method='zscore', **kw),
        'Stats_IQR': lambda **kw: StatsDetector(method='iqr', **kw),
    }
    DETECTOR_NAMES = list(DETECTOR_MAP.keys())

    dd_state = gr.State({'loader': None, 'detector': None, 'X': None,
                          'labels': None, 'scores': None, 'threshold': None})

    gr.Markdown("## 🔍 DeepDetect — 异常检测")
    gr.Markdown("上传时序 CSV，选择检测方法，自动识别异常点。支持 **7种检测器** 和有/无监督双模式。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 数据加载")
            dd_file = gr.File(label="上传 CSV", file_types=[".csv"])
            dd_label_col = gr.Textbox(label="标签列名（可选，有标签时计算 Precision/Recall）", placeholder="如: label")
            dd_load_btn = gr.Button("加载数据", variant="primary")
            dd_preview = gr.DataFrame(label="预览（前30行）", interactive=False)
            dd_info = gr.Markdown("请上传 CSV...")

        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ 检测配置")
            dd_target = gr.Dropdown(label="目标列（留空=全部数值列）", choices=[], value=None)
            dd_method = gr.Dropdown(
                label="检测方法",
                choices=DETECTOR_NAMES,
                value='IsolationForest'
            )
            dd_contamination = gr.Slider(0.01, 0.5, value=0.1, step=0.01,
                                         label="contamination（预期异常比例）")
            with gr.Row():
                dd_epochs = gr.Number(label="Epochs(AE)", value=100, visible=True)
                dd_n_est = gr.Number(label="n_estimators(IF)", value=100, visible=False)
            with gr.Row():
                dd_thresh_mode = gr.Radio(["auto", "custom"], value="auto", label="阈值模式")
                dd_custom_thresh = gr.Number(label="自定义阈值", value=0.5, visible=False)
            dd_detect_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")

    with gr.Row():
        dd_metrics = gr.JSON(label="📊 检测指标")
        dd_plot = gr.Plot(label="📉 检测结果可视化")

    with gr.Row():
        dd_export_btn = gr.Button("📤 导出结果 CSV", variant="secondary")
        dd_export_file = gr.File(label="下载结果")
        dd_download_plot_btn = gr.Button("📥 下载图表 PNG", variant="secondary")
        dd_download_plot_file = gr.File(label="下载图表")
        dd_status = gr.Markdown("")

    # ===== 事件绑定 =====
    def dd_on_load(file_obj, label_col_name):
        if file_obj is None:
            return gr.update(), "请上传文件", gr.update(), dd_state
        path = _extract_path(file_obj)
        if not path:
            return gr.update(), "无法读取文件", gr.update(), dd_state
        try:
            dl = DDLoader()
            dl.load_csv(path)
            if label_col_name and label_col_name in dl.df.columns:
                dl.set_label_column(label_col_name)
            numeric = dl.get_numeric_columns()
            preview = dl.get_preview(30)
            info = f"**已加载**：{dl.df.shape[0]}行 × {dl.df.shape[1]}列\n**数值列**：{len(numeric)}"
            new_state = {'loader': dl, 'detector': None, 'X': None,
                         'labels': None, 'scores': None, 'threshold': None}
            return gr.update(choices=numeric, value=numeric[0] if numeric else None), \
                   info, preview, new_state
        except Exception as e:
            return gr.update(), f"加载失败: {e}", gr.update(), dd_state

    dd_load_btn.click(dd_on_load, inputs=[dd_file, dd_label_col],
                      outputs=[dd_target, dd_info, dd_preview, dd_state])

    def dd_show_params(method, contamination):
        vis_ep = method in ('Autoencoder', 'LSTM')
        vis_est = method == 'IsolationForest'
        return gr.update(visible=vis_ep), gr.update(visible=vis_est)

    dd_method.change(dd_show_params, inputs=[dd_method, dd_contamination],
                     outputs=[dd_epochs, dd_n_est])

    def dd_on_detect(target_col, method, contamination, thresh_mode,
                     custom_thresh, epochs, n_est, state):
        loader = state['loader']
        if loader is None:
            return {}, None, state, "❌ 请先加载数据"
        try:
            X, y = loader.get_all_numeric_data(handle_missing='mean')
            if target_col and target_col in X.columns:
                feature_cols = [c for c in X.columns if c != target_col]
                X = X[feature_cols] if feature_cols else X[[target_col]]

            params = {'contamination': contamination}
            if method == 'Autoencoder':
                params['epochs'] = int(epochs)
            elif method == 'LSTM':
                params['epochs'] = int(epochs)
            elif method == 'IsolationForest':
                params['n_estimators'] = int(n_est)

            detector = DETECTOR_MAP[method](**params)
            detector.fit(X.values)
            scores = detector.score_samples(X.values)
            threshold = custom_thresh if thresh_mode == 'custom' else detector.get_threshold()
            labels = (scores > threshold).astype(int)

            y_true = y.values if y is not None else None
            eval_results = evaluate_detector(y_true, labels, scores)
            metrics_text = format_metrics_table(eval_results)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            idx = np.arange(len(X))

            # 时序图
            ax = axes[0, 0]
            vals = X.values[:, 0] if X.shape[1] > 1 else X.values.flatten()
            ax.plot(idx, vals, 'b-', alpha=0.6)
            anom_idx = np.where(labels == 1)[0]
            if len(anom_idx) > 0:
                ax.scatter(anom_idx, vals[anom_idx], c='red', s=30, zorder=5,
                           label=f'异常 ({len(anom_idx)})')
            ax.set_xlabel('Index')
            ax.set_title('时序异常检测')
            ax.legend()
            ax.grid(alpha=0.3)

            # 分数图
            ax = axes[0, 1]
            ax.plot(idx, scores, 'b-', alpha=0.5)
            ax.axhline(threshold, color='r', linestyle='--', label=f'阈值 {threshold:.3f}')
            ax.set_xlabel('Index')
            ax.set_title('异常分数')
            ax.legend()
            ax.grid(alpha=0.3)

            # 分布直方图
            ax = axes[1, 0]
            ax.hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(threshold, color='r', linestyle='--', linewidth=2)
            ax.set_title('分数分布')
            ax.grid(alpha=0.3)

            # 箱线图
            ax = axes[1, 1]
            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]
            bp = ax.boxplot([normal_scores, anomaly_scores],
                           labels=['正常', '异常'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax.set_title('分数对比')
            ax.grid(alpha=0.3)
            plt.tight_layout()

            new_state = dict(state)
            new_state.update({'detector': detector, 'X': X, 'labels': labels,
                             'scores': scores, 'threshold': threshold,
                             'eval_results': eval_results})

            # 保存图表为 PNG 供下载
            import tempfile as _tempfile, os as _os
            _plot_path = _os.path.join(_tempfile.gettempdir(), f"dd_plot_{int(time.time())}.png")
            fig.savefig(_plot_path, format='png', dpi=150, bbox_inches='tight')
            new_state['plot_path'] = _plot_path
            return eval_results, fig, new_state, metrics_text
        except Exception as e:
            import traceback
            return {}, None, state, f"❌ 检测失败: {e}\n```\n{traceback.format_exc()}\n```"

    dd_detect_btn.click(dd_on_detect,
                        inputs=[dd_target, dd_method, dd_contamination,
                                dd_thresh_mode, dd_custom_thresh, dd_epochs, dd_n_est, dd_state],
                        outputs=[dd_metrics, dd_plot, dd_state, dd_status])

    def dd_on_export(state):
        loader = state['loader']
        X = state['X']
        labels = state['labels']
        scores = state['scores']
        if loader is None or X is None or labels is None:
            return None, "❌ 没有可导出的结果"
        try:
            result_df = loader.export_with_labels(X, labels, scores)

            # 使用临时文件（与 DeepDetect/app.py 的 840a3b1 修复保持一致）
            import tempfile, os, time
            filename = f"anomaly_results_{int(time.time())}.csv"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            result_df.to_csv(filepath, index=False, encoding='utf-8')

            # 验证文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            file_size = os.path.getsize(filepath)

            return filepath, f"✅ 已导出 {len(result_df)} 行到 {filename}（{file_size} bytes）"
        except Exception as e:
            import traceback
            return None, f"❌ 导出失败: {str(e)}\n{traceback.format_exc()}"

    dd_export_btn.click(dd_on_export, inputs=[dd_state], outputs=[dd_export_file, dd_status])

    def dd_on_download_plot(state):
        plot_path = state.get('plot_path')
        if not plot_path or not os.path.exists(plot_path):
            return None, "❌ 请先运行检测生成图表"
        return plot_path, f"✅ 图表已保存"

    dd_download_plot_btn.click(dd_on_download_plot, inputs=[dd_state],
                              outputs=[dd_download_plot_file, dd_status])

    return dd_state


def build_deep_detect_ui():
    with gr.Tab("🔍 DeepDetect"):
        try:
            _build_deep_detect_ui()
        except Exception as e:
            gr.Markdown(f"❌ DeepDetect 加载失败: {e}")
            gr.Markdown("请确认 DeepDetect 依赖已安装：`pip install -r DeepDetect/requirements.txt`")


# ============================================================
# ========== 工具函数 =====
# ============================================================

def _extract_path(file_obj):
    """从 Gradio File 组件提取文件路径"""
    if file_obj is None:
        return None
    # Gradio 6.x ListFiles
    if hasattr(file_obj, 'root'):
        file_obj = file_obj.root
    if isinstance(file_obj, list):
        if len(file_obj) == 0:
            return None
        file_obj = file_obj[0]
    if isinstance(file_obj, dict):
        return file_obj.get('path')
    if hasattr(file_obj, 'path'):
        return str(file_obj.path)
    if isinstance(file_obj, str):
        return file_obj
    return None


def _plot_results(predictor, X_df, y_series, y_col):
    """绘制预测结果图 - 使用 PredictVisualizer 子刊风格"""
    try:
        preds = predictor.predict(X_df)
        y_true = y_series.values

        # 长度对齐（处理 LSTM/PatchTST 等序列模型输出长度与原始数据不一致的问题）
        min_len = min(len(y_true), len(preds))
        if min_len < len(y_true) or min_len < len(preds):
            logger.warning(f"预测长度不匹配: y_true={len(y_true)}, preds={len(preds)}，截断到 {min_len}")
            y_true = y_true[:min_len]
            preds = preds[:min_len]

        # 使用新的 PredictVisualizer
        viz = PredictVisualizer(figsize=(12, 5))

        if predictor.task_type == 'classification':
            # 分类任务：使用散点图 + 混淆矩阵
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            # 散点图
            axes[0].scatter(range(len(y_true)), y_true, alpha=0.5, label='Actual', s=20, c='#1f77b4')
            axes[0].scatter(range(len(preds)), preds, alpha=0.5, label='Predicted', s=20, c='#d62728')
            axes[0].set_xlabel('Sample')
            axes[0].set_ylabel(y_col)
            axes[0].legend()
            axes[0].set_title(f'{y_col} Actual vs Predicted')
            axes[0].grid(True, alpha=0.3)

            # 混淆矩阵
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            labels = sorted(y_series.unique())
            cm = confusion_matrix(y_series, preds, labels=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                       xticklabels=labels, yticklabels=labels)
            axes[1].set_title('Confusion Matrix')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            plt.tight_layout()
            return fig
        else:
            # 回归任务：使用子刊风格预测图 + 残差图
            # 自动选择放大区域（中间 20%）
            n = len(y_true)
            zoom_start = int(n * 0.4)
            zoom_end = int(n * 0.6)

            # 生成子刊风格时序图
            fig = viz.plot_prediction_timeseries(
                y_true, preds,
                title=f"{y_col} Prediction vs Actual",
                ylabel=y_col,
                zoom_range=(zoom_start, zoom_end),
                zoom_title=f"Zoomed Detail ({zoom_start}-{zoom_end})",
                labels={'true': 'Actual', 'pred': 'Predicted', 'ci': '95% CI'},
                show_metrics=True,
                figsize=(14, 8),
                dpi=150
            )
            return fig
    except Exception as e:
        logger.warning(f"绘图失败 (使用备用方案): {e}")
        # 降级备用方案
        try:
            import traceback
            traceback.print_exc()
            preds = predictor.predict(X_df)
            y_t = y_series.values
            p_t = preds
            min_len = min(len(y_t), len(p_t))
            y_t = y_t[:min_len]
            p_t = p_t[:min_len]
            n = len(y_t)
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].plot(range(n), y_t, label='Actual', alpha=0.7)
            axes[0].plot(range(n), p_t, label='Predicted', alpha=0.7)
            axes[0].legend()
            axes[0].set_title(f'{y_col} Prediction')
            axes[1].hist(y_t - p_t, bins=30, alpha=0.7)
            axes[1].set_title('Residuals')
            plt.tight_layout()
            return fig
        except Exception as e2:
            logger.error(f"备用绘图也失败: {e2}")
            return None


# ============================================================
# ========== 主程序 =====
# ============================================================

def main():
    with gr.Blocks(title="Deep-Research 统一平台") as app:
        # 顶栏
        gr.Markdown("""
        <div style="text-align:center; padding: 1.5em 0;">
            <h1 style="font-size: 2.2em; color: #818cf8; margin: 0;">🧠 Deep-Research 统一平台</h1>
            <p style="color: #94a3b8; font-size: 1.1em; margin: 0.5em 0 0 0;">
                时序预测 · 信号分类 · 异常检测 — 三大 AI 分析模块
            </p>
            <p style="color: #64748b; font-size: 0.85em; margin: 0.3em 0 0 0;">
                Powered by PyTorch · scikit-learn · SHAP
            </p>
        </div>
        """)

        # 三个 Tab
        build_deep_predict_ui()
        build_deep_classify_ui()
        build_deep_detect_ui()

        # 底栏
        gr.Markdown("""
        <div style="text-align:center; padding: 1em; color: #64748b; font-size: 0.8em;">
            Deep-Research v2.0 · © 2026 · Built with OpenClaw + Gradio
        </div>
        """)

    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        theme=gr.themes.Glass(),
        css=CSS
    )


if __name__ == "__main__":
    main()
