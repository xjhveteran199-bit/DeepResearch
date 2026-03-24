# -*- coding: utf-8 -*-
"""
DeepClassify - Gradio Web 界面
信号分类模块
"""

import os
import sys
import logging
from pathlib import Path

# UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

import gradio as gr
import pandas as pd
import numpy as np

# 添加 src 路径
APP_ROOT = Path(__file__).parent
SRC_ROOT = APP_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(APP_ROOT))

# ============ 日志配置 ============
log_dir = APP_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "deepclassify.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============ 全局状态 ============
class AppState:
    def __init__(self):
        self.data_loader = None
        self.classifier = None
        self.X_features = None
        self.y_target = None
        self.label_encoder = None
        self.class_names = []
        self.train_result = None
        self.metrics = {}
        self.predict_df = None  # 待预测数据
        self.model_save_path = None

    def reset(self):
        self.__init__()


state = AppState()


# ============ 辅助函数 ============
def extract_file_path(file_obj):
    """从 Gradio File 组件提取文件路径"""
    if file_obj is None:
        return None
    if isinstance(file_obj, list) and len(file_obj) > 0:
        file_obj = file_obj[0]
    if hasattr(file_obj, 'path'):
        return str(file_obj.path)
    if isinstance(file_obj, dict):
        return str(file_obj.get('path', ''))
    if isinstance(file_obj, str):
        return file_obj
    return None


# ============ Tab 1: 数据加载 ============
def on_upload_csv(file_obj):
    """上传并加载 CSV"""
    path = extract_file_path(file_obj)
    if not path:
        return gr.update(), "❌ 未找到文件"

    from src.core.data_loader import DataLoader
    loader = DataLoader()
    ok, msg = loader.load_csv(path)

    if not ok:
        return gr.update(), msg

    state.data_loader = loader
    summary = loader.get_summary()

    # 预览
    preview_df = loader.get_preview(20)
    preview_html = f"<pre>{preview_df.to_string()}</pre>"

    info_lines = [
        f"✅ {msg}",
        f"",
        f"📊 数据形状: {summary['shape'][0]} 行 × {summary['shape'][1]} 列",
        f"📌 数值列: {len(summary['numeric_cols'])}",
        f"📌 类别列: {len(summary['categorical_cols'])}",
        f"📌 日期列: {len(summary['date_cols'])}",
    ]

    cols_dropdown = gr.update(
        choices=list(loader.df.columns),
        value=None
    )
    return cols_dropdown, "\n".join(info_lines), preview_html


# ============ Tab 2: 特征/标签选择 ============
def on_target_select(target_col, test_size):
    """选择目标列后，更新类别信息和特征预览"""
    if state.data_loader is None:
        return gr.update(), gr.update(), "❌ 请先加载数据"

    loader = state.data_loader
    ok, msg = loader.select_target(target_col)
    if not ok:
        return gr.update(), gr.update(), msg

    # 类别分布
    dist = loader.get_class_distribution()
    class_info = "\n".join([f"  {k}: {v} 样本" for k, v in dist.items()])

    # 特征矩阵
    state.y_target = loader.get_target()
    state.X_features = loader.get_feature_matrix(exclude_cols=[target_col])

    n_features = state.X_features.shape[1]
    n_samples = len(state.X_features)

    info = (
        f"✅ 目标列: {target_col}\n"
        f"📊 类别分布:\n{class_info}\n\n"
        f"📊 特征数: {n_features}, 样本数: {n_samples}\n"
        f"📊 训练/测试划分比例: {(1-float(test_size)):.0%} / {float(test_size):.0%}"
    )

    return gr.update(choices=list(state.X_features.columns)), gr.update(), info


# ============ Tab 3: 模型选择 ============
def on_model_select(model_name):
    """选择模型后更新参数面板"""
    if model_name == "CNN1D":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif model_name == "RandomForest":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif model_name == "GradientBoosting":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    elif model_name == "SVM":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


# ============ Tab 4: 训练 ============
def on_train(model_name, test_size,
             # CNN1D params
             cnn_hidden, cnn_kernel, cnn_epochs, cnn_lr, cnn_bs,
             # RF params
             rf_trees, rf_depth,
             # GB params
             gb_trees, gb_depth, gb_lr,
             # SVM params
             svm_c, svm_kernel):
    """执行训练"""
    if state.data_loader is None:
        return "❌ 请先加载数据", gr.update(), gr.update(), gr.update(), gr.update()

    if state.X_features is None or state.y_target is None:
        return "❌ 请选择特征和目标列", gr.update(), gr.update(), gr.update(), gr.update()

    # 获取编码后的标签
    y_enc, le = state.data_loader.get_target_encoded()
    state.label_encoder = le
    state.class_names = list(le.classes_)

    # 导入分类器
    from src.models import CNN1DClassifyWrapper, RFClassifier, GBClassifier, SVMClassifier

    # 构造参数字典
    params = {}
    if model_name == "CNN1D":
        params = {
            'hidden_channels': int(cnn_hidden),
            'kernel_size': int(cnn_kernel),
            'epochs': int(cnn_epochs),
            'learning_rate': float(cnn_lr),
            'batch_size': int(cnn_bs),
        }
        classifier = CNN1DClassifyWrapper(**params)
    elif model_name == "RandomForest":
        params = {
            'n_estimators': int(rf_trees),
            'max_depth': int(rf_depth) if rf_depth > 0 else None,
        }
        classifier = RFClassifier(**params)
    elif model_name == "GradientBoosting":
        params = {
            'n_estimators': int(gb_trees),
            'max_depth': int(gb_depth),
            'learning_rate': float(gb_lr),
        }
        classifier = GBClassifier(**params)
    elif model_name == "SVM":
        params = {
            'C': float(svm_c),
            'kernel': str(svm_kernel),
        }
        classifier = SVMClassifier(**params)
    else:
        return f"❌ 未知模型: {model_name}", gr.update(), gr.update(), gr.update(), gr.update()

    state.classifier = classifier

    # 训练
    ok, msg = classifier.fit(
        X_train=state.X_features,
        y_train=y_enc,
        X_val=None,
        y_val=None
    )

    if not ok:
        return f"❌ {msg}", gr.update(), gr.update(), gr.update(), gr.update()

    # 计算详细指标
    from sklearn.model_selection import train_test_split
    from src.core.metrics import ClassificationMetrics

    X_arr = state.X_features.values.astype(np.float32)
    y_arr = y_enc.values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y_arr, test_size=float(test_size), random_state=42
    )

    y_pred = classifier.predict(X_te)
    y_proba = classifier.predict_proba(X_te)

    mc = ClassificationMetrics()
    full_metrics = mc.compute(
        y_true=y_te.astype(int),
        y_pred=y_pred.astype(int) if y_pred.dtype != int else y_pred,
        y_proba=y_proba,
        labels=list(range(len(state.class_names)))
    )

    state.metrics = full_metrics

    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, y_pred)
    cm_html = _cm_to_html(cm, state.class_names)

    # ROC 曲线
    roc_fig = _plot_roc_curve(y_te, y_proba, state.class_names)

    # 指标展示
    metrics_html = _metrics_to_html(full_metrics)

    # 保存模型并提供下载
    save_path = str(APP_ROOT / "deepclassify_model.pkl")
    classifier.save(save_path)

    return msg, metrics_html, cm_html, roc_fig, save_path


def _cm_to_html(cm, labels):
    """混淆矩阵转 HTML 表格"""
    header = "<tr><th></th>" + "".join([f"<th>{l}</th>" for l in labels]) + "</tr>"
    rows = []
    for i, row in enumerate(cm):
        rows.append(
            "<tr>"
            f"<td><b>{labels[i]}</b></td>"
            + "".join([f"<td>{v}</td>" for v in row])
            + "</tr>"
        )
    table = (
        "<table style='border-collapse:collapse; width:100%;'>"
        f"<caption><b>混淆矩阵</b></caption>"
        + header
        + "".join(rows)
        + "</table>"
    )
    return f"<div style='overflow-x:auto;'>{table}</div>"


def _metrics_to_html(metrics):
    """指标转 HTML"""
    rows = []
    for k, v in metrics.items():
        if isinstance(v, float):
            rows.append(f"<tr><td><b>{k}</b></td><td>{v:.4f}</td></tr>")
        else:
            rows.append(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>")
    return (
        "<table style='border-collapse:collapse; width:100%;'>"
        + "<caption><b>评估指标</b></caption>"
        + "".join(rows)
        + "</table>"
    )


def _plot_roc_curve(y_true, y_proba, class_names):
    """绘制 ROC 曲线"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    n_classes = y_proba.shape[1]
    fig, ax = plt.subplots(figsize=(6, 5))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    else:
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            label = class_names[i] if i < len(class_names) else f"Class {i}"
            ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def on_download_model():
    """下载模型"""
    if state.classifier is None:
        return None
    save_path = APP_ROOT / "deepclassify_model.pkl"
    state.classifier.save(str(save_path))
    state.model_save_path = str(save_path)
    return str(save_path)


# ============ Tab 5: 预测新数据 ============
def on_upload_predict(file_obj):
    """上传待预测数据"""
    path = extract_file_path(file_obj)
    if not path:
        return "❌ 未找到文件", "⚠️ 请上传文件"

    try:
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            df = pd.read_csv(path, encoding='utf-8', errors='replace')

        state.predict_df = df
        preview = df.head(10).to_string()
        info = f"✅ 加载成功: {df.shape[0]} 行 × {df.shape[1]} 列\n\n```\n{preview}\n```"
        return info, info  # 加载信息和预览

    except Exception as e:
        return f"❌ 加载失败: {str(e)}", f"❌ 加载失败: {str(e)}"


def on_predict_execute():
    """执行预测"""
    if state.classifier is None:
        return "❌ 模型未训练，请先训练模型", gr.update(visible=False)

    if state.predict_df is None:
        return "⚠️ 请先上传待预测CSV文件", gr.update(visible=False)

    try:
        # 使用与训练时相同的特征处理
        loader = state.data_loader
        target_col = loader.df.columns[0]  # 目标列名（用户选择的）

        # 获取特征（与训练相同的处理）
        X_pred = loader.get_feature_matrix(exclude_cols=[target_col])

        # 如果预测数据有额外的列，使用前 n_features 列
        if state.predict_df.shape[1] > X_pred.shape[1]:
            # 尝试对齐列
            pred_cols = state.predict_df.columns.tolist()
            feat_cols = state.X_features.columns.tolist()
            # 找共同的列
            common = [c for c in feat_cols if c in pred_cols]
            if common:
                X_pred = state.predict_df[common].values.astype(np.float32)
            else:
                # 使用前 n_features 列
                X_pred = state.predict_df.iloc[:, :state.X_features.shape[1]].values.astype(np.float32)
        else:
            X_pred = state.predict_df.iloc[:, :state.X_features.shape[1]].values.astype(np.float32)

        # 预测
        y_pred = state.classifier.predict(X_pred)
        y_proba = state.classifier.predict_proba(X_pred)

        # 结果
        result_df = pd.DataFrame({
            '预测类别': y_pred,
            '预测概率': y_proba.max(axis=1).round(4)
        })

        result_html = result_df.head(50).to_string()
        result_text = f"预测完成！共 {len(y_pred)} 条数据\n\n{result_html}"

        # 保存 CSV
        save_path = APP_ROOT / "predictions.csv"
        result_df.to_csv(save_path, index=False, encoding='utf-8-sig')

        return result_text, gr.update(value=str(save_path), visible=True)

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return f"❌ 预测失败: {str(e)}", gr.update(visible=False)


# ============ 构建 Gradio 界面 ============
def build_ui():
    with gr.Blocks(title="DeepClassify - 信号分类", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔬 DeepClassify - AI 信号分类模块")
        gr.Markdown("上传 CSV（特征 + 标签），选择分类模型，训练并预测。")

        with gr.Tabs():
            # ===== Tab 1: 数据加载 =====
            with gr.TabItem("📂 数据加载"):
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            label="上传 CSV 文件",
                            file_types=[".csv"],
                            type="filepath"
                        )
                        load_btn = gr.Button("加载数据", variant="primary")
                    with gr.Column():
                        data_info = gr.Textbox(label="数据信息", lines=8, interactive=False)
                preview_html = gr.HTML(label="数据预览")

                load_btn.click(
                    fn=on_upload_csv,
                    inputs=[file_upload],
                    outputs=[gr.Dropdown(label="选择目标列（标签列）"), data_info, preview_html]
                )

            # ===== Tab 2: 特征/标签选择 =====
            with gr.TabItem("⚙️ 特征与标签"):
                with gr.Row():
                    with gr.Column(scale=1):
                        target_col = gr.Dropdown(label="目标列（标签）", choices=[], allow_empty_value=False)
                        test_size = gr.Slider(0.05, 0.4, value=0.2, step=0.05, label="测试集比例")
                        select_btn = gr.Button("确认选择", variant="primary")
                    with gr.Column(scale=2):
                        feature_col = gr.Dropdown(
                            label="特征列（默认使用全部数值列）",
                            choices=[],
                            allow_empty_value=True,
                            multiselect=True
                        )
                        select_info = gr.Textbox(label="选择信息", lines=10, interactive=False)

                select_btn.click(
                    fn=on_target_select,
                    inputs=[target_col, test_size],
                    outputs=[feature_col, target_col, select_info]
                )

            # ===== Tab 3: 模型选择 =====
            with gr.TabItem("🤖 选择模型"):
                model_name = gr.Radio(
                    ["CNN1D", "RandomForest", "GradientBoosting", "SVM"],
                    label="分类模型",
                    value="RandomForest"
                )
                gr.Markdown("---")
                gr.Markdown("**模型参数**")

                # CNN1D 参数
                with gr.Group(visible=False) as cnn_group:
                    gr.Markdown("#### CNN1D 参数")
                    with gr.Row():
                        cnn_hidden = gr.Number(label="隐藏通道", value=64)
                        cnn_kernel = gr.Number(label="卷积核大小", value=3)
                        cnn_epochs = gr.Number(label="训练轮次", value=50)
                    with gr.Row():
                        cnn_lr = gr.Number(label="学习率", value=0.001)
                        cnn_bs = gr.Number(label="批大小", value=32)

                # RF 参数
                with gr.Group(visible=False) as rf_group:
                    gr.Markdown("#### RandomForest 参数")
                    with gr.Row():
                        rf_trees = gr.Number(label="树数量", value=100)
                        rf_depth = gr.Number(label="最大深度 (0=不限)", value=10)

                # GB 参数
                with gr.Group(visible=False) as gb_group:
                    gr.Markdown("#### GradientBoosting 参数")
                    with gr.Row():
                        gb_trees = gr.Number(label="树数量", value=100)
                        gb_depth = gr.Number(label="最大深度", value=5)
                        gb_lr = gr.Number(label="学习率", value=0.1)

                # SVM 参数
                with gr.Group(visible=False) as svm_group:
                    gr.Markdown("#### SVM 参数")
                    with gr.Row():
                        svm_c = gr.Number(label="C (正则化)", value=1.0)
                        svm_kernel = gr.Dropdown(
                            label="核函数",
                            choices=['rbf', 'linear', 'poly', 'sigmoid'],
                            value='rbf'
                        )

                model_name.change(
                    fn=on_model_select,
                    inputs=[model_name],
                    outputs=[cnn_group, rf_group, gb_group, svm_group]
                )

            # ===== Tab 4: 训练与评估 =====
            with gr.TabItem("📊 训练与评估"):
                with gr.Row():
                    with gr.Column():
                        train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
                        download_file = gr.File(label="💾 下载模型文件", visible=False)
                    with gr.Column():
                        train_msg = gr.Textbox(label="训练结果", lines=5, interactive=False)

                gr.Markdown("---")
                gr.Markdown("**评估结果**")
                with gr.Row():
                    with gr.Column(scale=1):
                        metrics_display = gr.HTML(label="指标")
                    with gr.Column(scale=1):
                        cm_display = gr.HTML(label="混淆矩阵")
                roc_plot = gr.Plot(label="ROC 曲线")

                train_btn.click(
                    fn=on_train,
                    inputs=[
                        model_name, test_size,
                        cnn_hidden, cnn_kernel, cnn_epochs, cnn_lr, cnn_bs,
                        rf_trees, rf_depth,
                        gb_trees, gb_depth, gb_lr,
                        svm_c, svm_kernel
                    ],
                    outputs=[train_msg, metrics_display, cm_display, roc_plot, download_file]
                )

                download_file.change(
                    fn=on_download_model,
                    inputs=[],
                    outputs=[download_file]
                )

            # ===== Tab 5: 预测新数据 =====
            with gr.TabItem("🔮 预测新数据"):
                gr.Markdown("上传 CSV 文件（特征列需与训练数据一致），进行批量预测。")
                with gr.Row():
                    with gr.Column():
                        predict_file = gr.File(label="待预测 CSV", file_types=[".csv"])
                        predict_btn = gr.Button("📂 加载文件", variant="secondary")
                        predict_execute_btn = gr.Button("🔮 执行预测", variant="primary")
                        predict_load_info = gr.Textbox(label="文件信息", lines=3, interactive=False)
                    with gr.Column():
                        predict_result = gr.Textbox(label="预测结果", lines=10, interactive=False)
                        predict_download = gr.File(label="下载预测结果", visible=False)

                predict_btn.click(
                    fn=on_upload_predict,
                    inputs=[predict_file],
                    outputs=[predict_load_info, predict_result]
                )

                predict_execute_btn.click(
                    fn=on_predict_execute,
                    inputs=[],
                    outputs=[predict_result, predict_download]
                )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
