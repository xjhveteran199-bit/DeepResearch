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
        return gr.update(), "❌ 未找到文件", "", gr.update()

    from src.core.data_loader import DataLoader
    loader = DataLoader()
    ok, msg = loader.load_csv(path)

    if not ok:
        return gr.update(), msg, "", gr.update()

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
    return cols_dropdown, "\n".join(info_lines), preview_html, gr.update(choices=list(loader.df.columns), value=None)


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
def on_train(model_name, test_size, kfold,
             # CNN1D params
             cnn_hidden, cnn_kernel, cnn_epochs, cnn_lr, cnn_bs,
             # RF params
             rf_trees, rf_depth,
             # GB params
             gb_trees, gb_depth, gb_lr, gb_backend,
             # SVM params
             svm_c, svm_kernel):
    """执行训练"""
    if state.data_loader is None:
        return "❌ 请先加载数据", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    if state.X_features is None or state.y_target is None:
        return "❌ 请选择特征和目标列", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

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
            'backend': str(gb_backend),
        }
        classifier = GBClassifier(**params)
    elif model_name == "SVM":
        params = {
            'C': float(svm_c),
            'kernel': str(svm_kernel),
        }
        classifier = SVMClassifier(**params)
    else:
        return f"❌ 未知模型: {model_name}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    state.classifier = classifier

    # 训练
    ok, msg = classifier.fit(
        X_train=state.X_features,
        y_train=y_enc,
        X_val=None,
        y_val=None
    )

    if not ok:
        return f"❌ {msg}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # 计算详细指标
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from src.core.metrics import ClassificationMetrics

    X_arr = state.X_features.values.astype(np.float32)
    y_arr = y_enc.values

    # K-Fold 交叉验证
    kfold_msg = ""
    kfold_html = ""
    if kfold > 1:
        skf = StratifiedKFold(n_splits=int(kfold), shuffle=True, random_state=42)
        fold_scores = {'Accuracy': [], 'F1_weighted': [], 'Precision_weighted': [], 'Recall_weighted': []}
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
            # 克隆分类器
            fold_clf = _clone_classifier(model_name, locals())
            fold_clf.fit(state.X_features.iloc[tr_idx], pd.Series(y_tr))
            y_pred_fold = fold_clf.predict(X_val)
            y_proba_fold = fold_clf.predict_proba(X_val)
            mc_fold = ClassificationMetrics()
            m_fold = mc_fold.compute(y_val, y_pred_fold, y_proba_fold, labels=list(range(len(state.class_names))))
            fold_scores['Accuracy'].append(m_fold.get('Accuracy', 0))
            fold_scores['F1_weighted'].append(m_fold.get('F1_weighted', 0))
            fold_scores['Precision_weighted'].append(m_fold.get('Precision_weighted', 0))
            fold_scores['Recall_weighted'].append(m_fold.get('Recall_weighted', 0))

        # K-Fold 均值
        kfold_lines = ["<table style='border-collapse:collapse; width:100%;'>",
                       "<caption><b>K-Fold 交叉验证结果 (K={})</b></caption>".format(kfold),
                       "<tr><th>指标</th><th>均值</th><th>标准差</th></tr>"]
        for metric, vals in fold_scores.items():
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            kfold_lines.append(f"<tr><td>{metric}</td><td>{mean_v:.4f}</td><td>{std_v:.4f}</td></tr>")
        kfold_lines.append("</table>")
        kfold_html = "\n".join(kfold_lines)
        kfold_msg = f"✅ {kfold}-Fold CV 完成"

    # 测试集评估
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
    
    # 混淆矩阵图片（使用新 visualizer）
    cm_fig = _plot_confusion_matrix(y_te, y_pred, state.class_names)

    # ROC 曲线
    roc_fig = _plot_roc_curve(y_te, y_proba, state.class_names)
    
    # t-SNE 可视化（仅当样本数足够时）
    tsne_fig = None
    if len(y_te) >= 10:  # t-SNE 需要足够样本
        try:
            tsne_fig = _plot_tsne(X_te, y_te, state.class_names)
        except Exception as e:
            logger.warning(f"t-SNE 生成失败: {e}")

    # 指标展示
    metrics_html = _metrics_to_html(full_metrics)

    # 保存模型并提供下载
    save_path = str(APP_ROOT / "deepclassify_model.pkl")
    classifier.save(save_path)

    return msg, metrics_html, cm_html, roc_fig, save_path, kfold_html, cm_fig, tsne_fig


def _clone_classifier(model_name, local_vars):
    """克隆分类器（用于K-Fold）"""
    from src.models import CNN1DClassifyWrapper, RFClassifier, GBClassifier, SVMClassifier
    if model_name == "CNN1D":
        return CNN1DClassifyWrapper(
            hidden_channels=int(local_vars.get('cnn_hidden', 64)),
            kernel_size=int(local_vars.get('cnn_kernel', 3)),
            epochs=int(local_vars.get('cnn_epochs', 10)),
            learning_rate=float(local_vars.get('cnn_lr', 0.001)),
            batch_size=int(local_vars.get('cnn_bs', 32)),
        )
    elif model_name == "RandomForest":
        rf_depth = int(local_vars.get('rf_depth', 10))
        return RFClassifier(
            n_estimators=int(local_vars.get('rf_trees', 100)),
            max_depth=rf_depth if rf_depth > 0 else None,
        )
    elif model_name == "GradientBoosting":
        return GBClassifier(
            n_estimators=int(local_vars.get('gb_trees', 100)),
            max_depth=int(local_vars.get('gb_depth', 5)),
            learning_rate=float(local_vars.get('gb_lr', 0.1)),
            backend=str(local_vars.get('gb_backend', 'auto')),
        )
    elif model_name == "SVM":
        return SVMClassifier(
            C=float(local_vars.get('svm_c', 1.0)),
            kernel=str(local_vars.get('svm_kernel', 'rbf')),
        )
    raise ValueError(f"Unknown model: {model_name}")


# ============ SHAP 分析 ============
def on_shar_analysis():
    """执行 SHAP 可解释性分析"""
    if state.classifier is None:
        return "❌ 请先训练模型", gr.update(visible=False)

    try:
        import shap
        X_arr = state.X_features.values.astype(np.float32)

        # 取少量样本作为背景
        n_bg = min(50, len(X_arr) // 2)
        background = X_arr[:n_bg]
        test_sample = X_arr[n_bg:n_bg + 10]

        # 获取预测函数
        def predict_fn(x):
            return state.classifier.predict_proba(x)

        model_type = type(state.classifier).__name__
        shap_html = ""
        shap_fig = None

        if model_type in ('RFClassifier', 'GBClassifier'):
            # 树模型使用 TreeExplainer
            explainer = shap.TreeExplainer(state.classifier._model)
            shap_values = explainer.shap_values(test_sample)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values
            # 绘制 beeswarm
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_vals, test_sample, feature_names=list(state.X_features.columns),
                             show=False, plot_size=None)
            shap_fig = fig
            # HTML 摘要
            feat_imp = state.classifier.get_feature_importance()
            rows = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
            rows_html = "".join([f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in rows])
            shap_html = (
                "<table style='border-collapse:collapse; width:100%;'>"
                "<caption><b>SHAP 特征重要性 (TreeExplainer)</b></caption>"
                "<tr><th>特征</th><th>SHAP 重要性</th></tr>"
                + rows_html + "</table>"
            )
        else:
            # 其他模型（CNN1D, SVM）使用 KernelExplainer
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(test_sample, nsamples=50)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values

            # 绘制 beeswarm
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            try:
                shap.summary_plot(shap_vals, test_sample,
                                  feature_names=list(state.X_features.columns),
                                  show=False, plot_size=None)
                shap_fig = fig
            except Exception:
                shap_fig = None

            # 简单的特征重要性（基于SHAP值绝对值的均值）
            mean_abs = np.abs(shap_vals).mean(axis=0)
            feat_names = list(state.X_features.columns)
            rows = sorted(zip(feat_names, mean_abs), key=lambda x: x[1], reverse=True)
            rows_html = "".join([f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in rows])
            shap_html = (
                "<table style='border-collapse:collapse; width:100%;'>"
                "<caption><b>SHAP 特征重要性 (KernelExplainer)</b></caption>"
                "<tr><th>特征</th><th>Mean |SHAP|</th></tr>"
                + rows_html + "</table>"
            )

        # 保存 SHAP 图
        if shap_fig is not None:
            import matplotlib.pyplot as plt
            shap_path = APP_ROOT / "shap_summary.png"
            shap_fig.savefig(shap_path, bbox_inches='tight', dpi=150)
            plt.close(shap_fig)
            shap_plot_html = f"<img src='file/{shap_path}' width='100%'/>"
        else:
            shap_plot_html = ""

        result = f"✅ SHAP 分析完成！\n模型: {model_type}\n样本: {len(test_sample)}"
        return result, gr.update(value=shap_plot_html, visible=True)

    except ImportError as e:
        return f"⚠️ SHAP 库未安装或版本不兼容: {str(e)[:100]}", gr.update(visible=False)
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return f"❌ SHAP 分析失败: {str(e)[:200]}", gr.update(visible=False)


# ============ 模型对比 ============
def on_compare_all(test_size, kfold):
    """训练并对比所有模型"""
    if state.data_loader is None:
        return "❌ 请先加载数据", gr.update()
    if state.X_features is None:
        return "❌ 请先选择特征和标签", gr.update()

    y_enc, le = state.data_loader.get_target_encoded()
    state.class_names = list(le.classes_)
    X_arr = state.X_features.values.astype(np.float32)
    y_arr = y_enc.values

    from src.models import CNN1DClassifyWrapper, RFClassifier, GBClassifier, SVMClassifier
    from sklearn.model_selection import StratifiedKFold
    from src.core.metrics import ClassificationMetrics

    models_config = [
        ("CNN1D", CNN1DClassifyWrapper(hidden_channels=64, kernel_size=3, epochs=30,
                                       learning_rate=0.001, batch_size=32)),
        ("RandomForest", RFClassifier(n_estimators=50, max_depth=10)),
        ("GradientBoosting(GB)", GBClassifier(n_estimators=50, max_depth=5,
                                               learning_rate=0.1, backend='sklearn')),
        ("SVM(RBF)", SVMClassifier(C=1.0, kernel='rbf')),
    ]

    results = []
    for name, clf in models_config:
        try:
            # K-Fold CV
            if kfold > 1:
                skf = StratifiedKFold(n_splits=int(kfold), shuffle=True, random_state=42)
                acc_list, f1_list = [], []
                for tr_idx, val_idx in skf.split(X_arr, y_arr):
                    fold_clf = _clone_from_name(name)
                    fold_clf.fit(state.X_features.iloc[tr_idx], pd.Series(y_arr[tr_idx]))
                    y_p = fold_clf.predict(X_arr[val_idx])
                    mc = ClassificationMetrics()
                    m = mc.compute(y_arr[val_idx], y_p,
                                   fold_clf.predict_proba(X_arr[val_idx]),
                                   labels=list(range(len(state.class_names))))
                    acc_list.append(m.get('Accuracy', 0))
                    f1_list.append(m.get('F1_weighted', 0))
                cv_acc = np.mean(acc_list)
                cv_f1 = np.mean(f1_list)
                cv_std = np.std(acc_list)
            else:
                cv_acc = cv_f1 = cv_std = 0.0

            # 最终训练 + 测试评估
            clf.fit(state.X_features, pd.Series(y_arr))
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_arr, y_arr, test_size=float(test_size), random_state=42)
            y_pred = clf.predict(X_te)
            y_proba = clf.predict_proba(X_te)
            mc = ClassificationMetrics()
            m = mc.compute(y_te, y_pred, y_proba,
                          labels=list(range(len(state.class_names))))

            results.append({
                'Model': name,
                'CV_Acc': f"{cv_acc:.4f}" if cv_acc else "-",
                'CV_F1': f"{cv_f1:.4f}" if cv_f1 else "-",
                'CV_Std': f"{cv_std:.4f}" if cv_std else "-",
                'Test_Acc': f"{m.get('Accuracy', 0):.4f}",
                'Test_F1': f"{m.get('F1_weighted', 0):.4f}",
                'Test_AUC': f"{m.get('ROC_AUC', m.get('ROC_AUC_micro', 0)):.4f}",
            })
        except Exception as e:
            results.append({
                'Model': name,
                'CV_Acc': 'ERROR', 'CV_F1': 'ERROR', 'CV_Std': 'ERROR',
                'Test_Acc': 'ERROR', 'Test_F1': 'ERROR', 'Test_AUC': 'ERROR',
                '_error': str(e)
            })

    # 生成对比 HTML 表格
    df_res = pd.DataFrame(results)
    # 排序（按 Test_F1 降序，排除 ERROR 行）
    try:
        df_res['_sort'] = df_res['Test_F1'].apply(
            lambda x: float(x) if x not in ('ERROR', '-') else -1)
        df_res = df_res.sort_values('_sort', ascending=False).drop('_sort', axis=1)
    except Exception:
        pass

    col_labels = ['Model', 'CV_Acc', 'CV_F1', 'CV_Std', 'Test_Acc', 'Test_F1', 'Test_AUC']
    headers = "<tr>" + "".join([f"<th>{c}</th>" for c in col_labels]) + "</tr>"
    rows_html = ""
    for _, row in df_res.iterrows():
        cells = "".join([f"<td>{row.get(c, '')}</td>" for c in col_labels])
        rows_html += f"<tr>{cells}</tr>"

    comparison_html = (
        "<table style='border-collapse:collapse; width:100%; font-size:13px;'>"
        "<caption><b>模型对比结果 (K={}, Test={})</b></caption>".format(kfold, test_size)
        + headers + rows_html + "</table>"
    )

    summary = f"✅ 模型对比完成！共 {len(results)} 个模型"
    return summary, gr.update(value=comparison_html, visible=True)


def _clone_from_name(name):
    """根据模型名称克隆分类器"""
    from src.models import CNN1DClassifyWrapper, RFClassifier, GBClassifier, SVMClassifier
    if name == "CNN1D":
        return CNN1DClassifyWrapper(hidden_channels=64, kernel_size=3, epochs=30,
                                    learning_rate=0.001, batch_size=32)
    elif name == "RandomForest":
        return RFClassifier(n_estimators=50, max_depth=10)
    elif "GB" in name:
        return GBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, backend='sklearn')
    elif "SVM" in name:
        return SVMClassifier(C=1.0, kernel='rbf')
    raise ValueError(f"Unknown model: {name}")


# ============ 辅助绘图函数 ============
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


def _plot_roc_curve(y_true, y_proba, class_names):
    """绘制 ROC 曲线（使用新 visualizer）"""
    from src.visualizer import ClassifyVisualizer
    viz = ClassifyVisualizer(dpi=150, figsize=(6, 5))
    fig, _ = viz.plot_roc_curve(y_true, y_proba, class_names)
    return fig


def _plot_confusion_matrix(y_true, y_pred, labels):
    """绘制混淆矩阵（使用新 visualizer）"""
    from src.visualizer import ClassifyVisualizer
    import matplotlib.pyplot as plt
    viz = ClassifyVisualizer(dpi=150, figsize=(6, 5))
    fig = viz.plot_confusion_matrix(y_true, y_pred, labels)
    return fig


def _plot_tsne(X_features, y_true, labels):
    """绘制 t-SNE 可视化（使用新 visualizer）"""
    from src.visualizer import ClassifyVisualizer
    import matplotlib.pyplot as plt
    viz = ClassifyVisualizer(dpi=150, figsize=(8, 6))
    fig = viz.plot_tsne(X_features, labels, y_true, perplexity=min(30, len(y_true)//4))
    return fig


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
    with gr.Blocks(title="DeepClassify - 信号分类") as app:
        gr.Markdown("# 🔬 DeepClassify - AI 信号分类模块")
        gr.Markdown("上传 CSV（特征 + 标签），选择分类模型，训练并预测。")

        # 预定义目标列组件（用于 Tab1 加载后更新 Tab2 的下拉框）
        target_col = gr.Dropdown(label="目标列（标签）", choices=[])

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

                file_upload.upload(
                    fn=on_upload_csv,
                    inputs=[file_upload],
                    outputs=[gr.Dropdown(label="选择目标列（标签列）"), data_info, preview_html, target_col]
                )
                load_btn.click(
                    fn=on_upload_csv,
                    inputs=[file_upload],
                    outputs=[gr.Dropdown(label="选择目标列（标签列）"), data_info, preview_html, target_col]
                )

            # ===== Tab 2: 特征/标签选择 =====
            with gr.TabItem("⚙️ 特征与标签"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # target_col 已在外部定义，此处复用
                        test_size = gr.Slider(0.05, 0.4, value=0.2, step=0.05, label="测试集比例")
                        select_btn = gr.Button("确认选择", variant="primary")
                    with gr.Column(scale=2):
                        feature_col = gr.Dropdown(
                            label="特征列（默认使用全部数值列）",
                            choices=[],
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
                    with gr.Row():
                        gb_backend = gr.Dropdown(
                            label="后端引擎",
                            choices=['auto', 'sklearn', 'lgbm', 'xgb'],
                            value='auto',
                            info="auto: 自动选择 (优先 XGBoost→LightGBM→sklearn)"
                        )

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
                        kfold = gr.Slider(1, 10, value=1, step=1,
                                          label="K-Fold 交叉验证 (1=禁用)", info="K=1表示不使用交叉验证")
                        compare_btn = gr.Button("⚖️ 对比所有模型", variant="secondary")
                        shap_btn = gr.Button("📊 SHAP 可解释性", variant="secondary")
                        download_file = gr.File(label="💾 下载模型文件", visible=False)
                    with gr.Column():
                        train_msg = gr.Textbox(label="训练结果", lines=5, interactive=False)
                        shap_msg = gr.Textbox(label="SHAP 结果", lines=3, interactive=False)

                gr.Markdown("---")
                gr.Markdown("**评估结果**")
                with gr.Row():
                    with gr.Column(scale=1):
                        metrics_display = gr.HTML(label="指标")
                    with gr.Column(scale=1):
                        cm_display = gr.HTML(label="混淆矩阵")
                with gr.Row():
                    with gr.Column(scale=1):
                        kfold_display = gr.HTML(label="K-Fold 结果", visible=True)
                    with gr.Column(scale=1):
                        shap_plot = gr.HTML(label="SHAP Summary Plot", visible=False)
                roc_plot = gr.Plot(label="ROC 曲线")
                cm_plot = gr.Plot(label="混淆矩阵图")
                tsne_plot = gr.Plot(label="t-SNE 可视化", visible=False)

                # 对比表格
                compare_display = gr.HTML(label="模型对比结果", visible=False)

                train_btn.click(
                    fn=on_train,
                    inputs=[
                        model_name, test_size, kfold,
                        cnn_hidden, cnn_kernel, cnn_epochs, cnn_lr, cnn_bs,
                        rf_trees, rf_depth,
                        gb_trees, gb_depth, gb_lr, gb_backend,
                        svm_c, svm_kernel
                    ],
                    outputs=[train_msg, metrics_display, cm_display, roc_plot, download_file, kfold_display, cm_plot, tsne_plot]
                )

                shap_btn.click(
                    fn=on_shar_analysis,
                    inputs=[],
                    outputs=[shap_msg, shap_plot]
                )

                compare_btn.click(
                    fn=on_compare_all,
                    inputs=[test_size, kfold],
                    outputs=[train_msg, compare_display]
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
