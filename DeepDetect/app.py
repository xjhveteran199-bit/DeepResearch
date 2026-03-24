"""
DeepDetect - 异常检测 Web 界面
基于 Gradio 构建
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from src.core.data_loader import DataLoader
from src.core.eval import evaluate_detector, format_metrics_table, find_anomaly_intervals
from src.visualizer import DetectVisualizer
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.ocsvm import OCSVMDetector
from src.models.lof_detector import LOFDetector
from src.models.stats_detector import StatsDetector
from src.models.lstm_detector import LSTMDetector

# 检测器注册表
DETECTOR_REGISTRY = {
    'IsolationForest': IsolationForestDetector,
    'Autoencoder': AutoencoderDetector,
    'OneClassSVM': OCSVMDetector,
    'LOF': LOFDetector,
    'LSTM': LSTMDetector,
    'Stats_ZScore': lambda **kw: StatsDetector(method='zscore', **kw),
    'Stats_IQR': lambda **kw: StatsDetector(method='iqr', **kw),
}

DETECTOR_LIST = list(DETECTOR_REGISTRY.keys())

# 全局状态
state = {
    'data_loader': None,
    'detector': None,
    'X': None,
    'y': None,
    'labels': None,
    'scores': None,
    'eval_results': None,
}


def load_data(file_obj, label_col):
    """加载CSV数据"""
    try:
        dl = DataLoader()
        if hasattr(file_obj, 'read'):
            bytes_data = file_obj.read()
            success, msg = dl.load_from_bytes(bytes_data, getattr(file_obj, 'name', 'upload.csv'))
        else:
            success, msg = dl.load_csv(file_obj)

        if not success:
            return None, msg, gr.update(), gr.update()

        state['data_loader'] = dl

        # 自动设置标签列
        if label_col and label_col in dl.df.columns:
            dl.set_label_column(label_col)

        numeric_cols = dl.get_numeric_columns()

        # 数据预览
        preview = dl.get_preview(30)

        # 数据统计
        summary = dl.get_summary()
        stats_md = f"""
**数据形状**: {summary['shape']}  
**数值列**: {len(numeric_cols)}  
**缺失值**: {len(summary['missing_info'])} 列有缺失值
"""
        if numeric_cols:
            stats_md += f"\n\n**数值列预览**:\n```\n{preview[numeric_cols].to_string(max_cols=6)}\n```"

        col_dropdown = gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None)

        return preview, stats_md, col_dropdown, gr.update()

    except Exception as e:
        return None, f"加载失败: {str(e)}", gr.update(), gr.update()


def detect_anomalies(target_col, detector_name, contamination, threshold_mode,
                      custom_threshold, z_threshold, iqr_factor, epochs, batch_size,
                      n_estimators, n_neighbors, seq_len, hidden_size, has_label):
    """执行异常检测"""
    try:
        dl = state['data_loader']
        if dl is None:
            return None, "请先加载数据", None, None

        # 获取数据
        X, y = dl.get_all_numeric_data(handle_missing='mean')

        if target_col and target_col in X.columns:
            # 使用指定列作为目标，其他列作为特征
            feature_cols = [c for c in X.columns if c != target_col]
            if feature_cols:
                X = X[feature_cols]
            else:
                X = X[[target_col]]

        if X.empty or X.shape[1] == 0:
            return None, "没有可用的数值特征", None, None

        # 构建检测器
        detector_cls = DETECTOR_REGISTRY.get(detector_name)
        if detector_cls is None:
            return None, f"未知检测器: {detector_name}", None, None

        # 参数
        params = {'contamination': contamination}
        if detector_name == 'Autoencoder':
            params.update({'epochs': epochs, 'batch_size': batch_size})
        elif detector_name == 'LSTM':
            params.update({'epochs': epochs, 'batch_size': batch_size, 'seq_len': int(seq_len), 'hidden_size': int(hidden_size)})
        elif detector_name == 'Stats_ZScore':
            params.update({'z_threshold': z_threshold})
        elif detector_name == 'Stats_IQR':
            params.update({'iqr_factor': iqr_factor})
        elif detector_name == 'IsolationForest':
            params.update({'n_estimators': n_estimators})
        elif detector_name == 'LOF':
            params.update({'n_neighbors': n_neighbors})

        detector = detector_cls(**params)
        detector.fit(X.values)

        # 预测
        scores = detector.score_samples(X.values)
        threshold = custom_threshold if threshold_mode == 'custom' else detector.get_threshold()
        labels = (scores > threshold).astype(int)

        state['detector'] = detector
        state['X'] = X
        state['y'] = y
        state['labels'] = labels
        state['scores'] = scores
        state['threshold'] = threshold

        # 评估
        y_true = y.values if y is not None else None
        eval_results = evaluate_detector(y_true, labels, scores)
        state['eval_results'] = eval_results

        metrics_text = format_metrics_table(eval_results)

        # 生成可视化
        fig = plot_results(X, scores, labels, threshold)

        return fig, metrics_text, gr.update(), gr.update()

    except Exception as e:
        import traceback
        return None, f"检测失败: {str(e)}\n{traceback.format_exc()}", None, None


def plot_results(X, scores, labels, threshold):
    """生成可视化图表 - 使用 DetectVisualizer 子刊风格"""
    n_samples = len(X)
    n_anomalies = int(np.sum(labels))
    
    # 使用 DetectVisualizer 生成图表
    # 时序数据
    if X.shape[1] == 1:
        times = np.arange(n_samples)
        values = X.values.flatten()
        col_name = X.columns[0]
    else:
        times = np.arange(n_samples)
        values = X.values[:, 0]
        col_name = X.columns[0]
    
    # 生成 2x2 图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 使用 DetectVisualizer 绘制异常时序标注图 (P0)
    idx = np.arange(n_samples)
    anomaly_idx = np.where(labels == 1)[0]
    
    # 子刊风格时序图
    ax1 = axes[0, 0]
    ax1.plot(idx, values, color='#1f77b4', linewidth=0.8, alpha=0.7, label='Time Series')
    if len(anomaly_idx) > 0:
        ax1.scatter(anomaly_idx, values[anomaly_idx], c='#d62728', s=25, zorder=5, 
                    label=f'Anomalies (n={n_anomalies})', alpha=0.8)
    ax1.set_xlabel('Index', fontsize=11)
    ax1.set_ylabel(col_name, fontsize=11)
    ax1.set_title('Time Series with Anomaly Detection', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. 异常分数时间序列 (带阈值线)
    ax2 = axes[0, 1]
    ax2.plot(idx, scores, color='#ff7f0e', linewidth=0.8, alpha=0.7, label='Anomaly Score')
    ax2.axhline(y=threshold, color='#d62728', linestyle='--', linewidth=1.5, 
                label=f'Threshold ({threshold:.3f})')
    if len(anomaly_idx) > 0:
        ax2.scatter(anomaly_idx, scores[anomaly_idx], c='#d62728', s=20, zorder=5, alpha=0.7)
    ax2.set_xlabel('Index', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Anomaly Scores Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. 使用 DetectVisualizer 绘制分数分布 (P0)
    ax3 = axes[1, 0]
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    ax3.hist(normal_scores, bins=40, alpha=0.6, color='steelblue', 
             label=f'Normal (n={len(normal_scores)})', edgecolor='white')
    ax3.hist(anomaly_scores, bins=40, alpha=0.6, color='lightcoral', 
             label=f'Anomaly (n={len(anomaly_scores)})', edgecolor='white')
    ax3.axvline(x=threshold, color='#d62728', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold:.3f})')
    ax3.set_xlabel('Anomaly Score', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Score Distribution', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 正常 vs 异常分数箱线图
    ax4 = axes[1, 1]
    data_to_plot = [normal_scores, anomaly_scores]
    bp = ax4.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax4.set_ylabel('Anomaly Score', fontsize=11)
    ax4.set_title('Score Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    stats_text = f'Samples: {n_samples} | Anomalies: {n_anomalies} ({n_anomalies/n_samples*100:.1f}%)'
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def plot_interval_visualization(X, scores, labels, threshold):
    """
    使用 DetectVisualizer 绘制异常区间可视化 (P1)
    
    Args:
        X: 特征数据 (DataFrame)
        scores: 异常分数
        labels: 预测标签
        threshold: 阈值
    
    Returns:
        matplotlib.figure.Figure
    """
    from src.visualizer import DetectVisualizer
    
    n_samples = len(X)
    
    # 获取时序数据
    if X.shape[1] == 1:
        times = np.arange(n_samples)
        values = X.values.flatten()
        col_name = X.columns[0]
    else:
        times = np.arange(n_samples)
        values = X.values[:, 0]
        col_name = X.columns[0]
    
    # 使用 find_anomaly_intervals 获取区间
    intervals = find_anomaly_intervals(labels)
    
    # 转换为 DataFrame
    interval_df = DetectVisualizer.create_interval_df(intervals)
    
    # 创建区间可视化
    fig = DetectVisualizer.plot_anomaly_intervals(
        times=times,
        values=values,
        interval_df=interval_df,
        labels=labels,
        title="Anomaly Intervals Visualization",
        ylabel=col_name,
        xlabel="Time Index",
        figsize=(14, 5),
        dpi=150,
        show_interval_labels=True,
        interval_color='#FFB6C1'
    )
    
    return fig


def export_results():
    """导出带标签的结果"""
    dl = state['data_loader']
    X = state['X']
    labels = state['labels']
    scores = state['scores']

    if dl is None or X is None or labels is None:
        return None, "没有可导出的结果"

    try:
        result_df = dl.export_with_labels(X, labels, scores)

        # 保存到临时CSV
        output_path = "anomaly_detection_results.csv"
        result_df.to_csv(output_path, index=False)

        return output_path, f"已导出 {len(result_df)} 行结果到 {output_path}"
    except Exception as e:
        return None, f"导出失败: {str(e)}"


def create_demo():
    """构建 Gradio 界面"""
    with gr.Blocks(title="DeepDetect - 异常检测") as demo:
        gr.Markdown("# 🔍 DeepDetect - 异常检测系统")
        gr.Markdown("上传时序数据，自动检测异常点。支持有标签和无监督两种模式。")

        with gr.Tabs():
            # ===== Tab 1: 数据加载 =====
            with gr.TabItem("📂 数据加载"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="上传 CSV 文件", file_types=[".csv"])
                        label_col_input = gr.Textbox(label="标签列名（可选，如有标签）", placeholder="如: label, target, is_anomaly")
                        load_btn = gr.Button("加载数据", variant="primary")

                    with gr.Column():
                        data_preview = gr.DataFrame(label="数据预览")
                        data_info = gr.Markdown("请上传CSV文件...")

                load_btn.click(
                    fn=load_data,
                    inputs=[file_input, label_col_input],
                    outputs=[data_preview, data_info]
                )

            # ===== Tab 2: 检测配置 =====
            with gr.TabItem("⚙️ 检测配置"):
                with gr.Row():
                    with gr.Column():
                        target_col = gr.Dropdown(label="目标检测列（留空则用全部数值列）", choices=[])
                        detector_name = gr.Dropdown(
                            label="检测方法",
                            choices=DETECTOR_LIST,
                            value=DETECTOR_LIST[0]
                        )
                        gr.Markdown("### 参数配置")
                        contamination = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="contamination（预期异常比例）")

                        with gr.Accordion("高级参数", open=False):
                            threshold_mode = gr.Radio(["auto", "custom"], value="auto", label="阈值模式")
                            custom_threshold = gr.Number(label="自定义阈值", value=0.5)
                            z_threshold = gr.Slider(1.0, 5.0, value=3.0, step=0.1, label="Z-score 阈值")
                            iqr_factor = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="IQR 因子")
                            epochs = gr.Slider(10, 200, value=50, step=10, label="训练轮数（Autoencoder/LSTM）")
                            batch_size = gr.Slider(8, 256, value=32, step=8, label="Batch Size")
                            n_estimators = gr.Slider(10, 200, value=100, step=10, label="IsolationForest 树数量")
                            n_neighbors = gr.Slider(5, 50, value=20, step=1, label="LOF 近邻数")
                            seq_len = gr.Slider(5, 100, value=20, step=5, label="LSTM 窗口长度（seq_len）")
                            hidden_size = gr.Slider(16, 256, value=64, step=16, label="LSTM 隐藏层大小")

                        detect_btn = gr.Button("开始检测", variant="primary", size="lg")

                    with gr.Column():
                        metrics_output = gr.Textbox(label="评估指标", lines=15, show_label=True)

                detect_btn.click(
                    fn=detect_anomalies,
                    inputs=[target_col, detector_name, contamination, threshold_mode,
                            custom_threshold, z_threshold, iqr_factor, epochs, batch_size,
                            n_estimators, n_neighbors, seq_len, hidden_size, gr.State()],
                    outputs=[gr.Plot(), metrics_output]
                )

            # ===== Tab 3: 可视化 =====
            with gr.TabItem("📊 检测结果"):
                gr.Markdown("### 时序图 & 异常标注")
                plot_output = gr.Plot(label="检测可视化")
                
            # ===== Tab 4: 异常区间 =====
            with gr.TabItem("🗺️ 异常区间"):
                gr.Markdown("### 异常区间可视化 (基于 find_anomaly_intervals)")
                interval_plot_output = gr.Plot(label="异常区间可视化")
                show_interval_btn = gr.Button("显示异常区间图", variant="secondary")
                
                def show_interval_plot():
                    """显示异常区间可视化"""
                    X = state.get('X')
                    scores = state.get('scores')
                    labels = state.get('labels')
                    threshold = state.get('threshold')
                    
                    if X is None or scores is None or labels is None or threshold is None:
                        return None
                    
                    try:
                        fig = plot_interval_visualization(X, scores, labels, threshold)
                        return fig
                    except Exception as e:
                        import traceback
                        print(f"Interval plot error: {str(e)}\n{traceback.format_exc()}")
                        return None
                
                show_interval_btn.click(
                    fn=show_interval_plot,
                    inputs=[],
                    outputs=[interval_plot_output]
                )

            # ===== Tab 5: 导出 =====
            with gr.TabItem("💾 导出结果"):
                gr.Markdown("### 导出带异常标签的数据")
                export_btn = gr.Button("导出 CSV", variant="primary")
                export_status = gr.Textbox(label="状态")
                export_file = gr.File(label="下载文件")

                export_btn.click(
                    fn=export_results,
                    inputs=[],
                    outputs=[export_file, export_status]
                )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
