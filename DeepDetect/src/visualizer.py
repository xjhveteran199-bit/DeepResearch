"""
DeepDetect 可视化模块 - 异常检测可视化
使用 scienceplots 实现子刊风格图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 尝试导入 scienceplots，如果失败则回退到标准样式
try:
    import scienceplots
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False
    import warnings
    warnings.warn("scienceplots not installed. Using default matplotlib style.")


class DetectVisualizer:
    """异常检测可视化器"""
    
    # 默认颜色配置
    COLORS = {
        'data': '#1f77b4',        # 蓝色 - 时序数据
        'anomaly': '#d62728',     # 红色 - 异常点
        'score': '#ff7f0e',       # 橙色 - 异常分数
        'threshold': '#d62728',   # 红色 - 阈值线
        'normal_bg': '#90EE90',   # 浅绿色 - 正常区间背景
        'anomaly_bg': '#FFB6C1',  # 浅红色 - 异常区间背景
    }
    
    @classmethod
    def _setup_style(cls):
        """设置图表样式"""
        if HAS_SCIENCEPLOTS:
            try:
                plt.style.use(['science', 'nature'])
            except Exception:
                plt.style.use('seaborn-v0_8-whitegrid')
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
        
        # 禁用 LaTeX 渲染（避免需要安装 LaTeX）
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    
    @classmethod
    def plot_anomaly_timeseries(
        cls,
        times,
        values,
        labels,
        scores,
        threshold,
        interval_df=None,
        save_path=None,
        title="Time Series Anomaly Detection",
        ylabel="Value",
        xlabel="Time Index",
        figsize=(12, 6),
        dpi=300,
        show_legend=True,
        show_grid=True,
        max_points=None
    ):
        """
        绘制异常时序标注图
        
        Args:
            times: 时间/索引数组
            values: 原始数据值
            labels: 预测标签 (0=正常, 1=异常)
            scores: 异常分数
            threshold: 阈值
            interval_df: 异常区间 DataFrame (可选，列: start, end, type)
            save_path: 保存路径
            title: 图表标题
            ylabel: Y轴标签
            xlabel: X轴标签
            figsize: 图表大小
            dpi: 分辨率
            show_legend: 是否显示图例
            show_grid: 是否显示网格
            max_points: 最大点数（用于大数据降采样）
        
        Returns:
            matplotlib.figure.Figure
        """
        cls._setup_style()
        
        # 数据降采样处理大数据
        if max_points and len(values) > max_points:
            step = len(values) // max_points
            indices = np.arange(0, len(values), step)
            times = np.array(times)[indices] if hasattr(times, '__getitem__') else indices
            values = np.array(values)[indices]
            labels = np.array(labels)[indices]
            scores = np.array(scores)[indices]
        
        # 创建图表 - 主坐标轴 + 次坐标轴
        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 主坐标轴：时序数据
        ax1.plot(times, values, color=cls.COLORS['data'], linewidth=1.0, 
                 alpha=0.8, label='Time Series', zorder=1)
        
        # 标注异常点
        anomaly_mask = labels == 1
        if np.any(anomaly_mask):
            ax1.scatter(times[anomaly_mask], values[anomaly_mask], 
                       c=cls.COLORS['anomaly'], s=30, zorder=5, 
                       label=f'Anomalies ({np.sum(anomaly_mask)})', alpha=0.8)
        
        ax1.set_xlabel(xlabel, fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12, color=cls.COLORS['data'])
        ax1.tick_params(axis='y', labelcolor=cls.COLORS['data'])
        
        # 次坐标轴：异常分数
        ax2 = ax1.twinx()
        ax2.plot(times, scores, color=cls.COLORS['score'], linewidth=1.0, 
                 alpha=0.7, linestyle='-', label='Anomaly Score', zorder=2)
        
        # 阈值线
        ax2.axhline(y=threshold, color=cls.COLORS['threshold'], 
                   linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f'Threshold ({threshold:.3f})', zorder=3)
        
        ax2.set_ylabel('Anomaly Score', fontsize=12, color=cls.COLORS['score'])
        ax2.tick_params(axis='y', labelcolor=cls.COLORS['score'])
        
        # 标题
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        
        # 图例
        if show_legend:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, 
                      loc='upper left', fontsize=10, framealpha=0.9)
        
        # 网格
        if show_grid:
            ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 调整布局
        fig.tight_layout()
        
        # 保存
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        return fig
    
    @classmethod
    def plot_anomaly_intervals(
        cls,
        times,
        values,
        interval_df,
        labels=None,
        save_path=None,
        title="Anomaly Intervals Visualization",
        ylabel="Value",
        xlabel="Time Index",
        figsize=(14, 5),
        dpi=300,
        show_grid=True,
        interval_color='#FFB6C1',
        normal_color='#E8F5E9',
        show_interval_labels=True,
        max_points=None
    ):
        """
        绘制异常区间可视化
        
        Args:
            times: 时间/索引数组
            values: 原始数据值
            interval_df: 异常区间 DataFrame，必须包含 start, end 列
            labels: 预测标签 (可选，用于标注异常点)
            save_path: 保存路径
            title: 图表标题
            ylabel: Y轴标签
            xlabel: X轴标签
            figsize: 图表大小
            dpi: 分辨率
            show_grid: 是否显示网格
            interval_color: 异常区间背景色
            normal_color: 正常区间背景色
            show_interval_labels: 是否显示区间起止时间标签
            max_points: 最大点数
        
        Returns:
            matplotlib.figure.Figure
        """
        cls._setup_style()
        
        # 数据降采样
        if max_points and len(values) > max_points:
            step = len(values) // max_points
            indices = np.arange(0, len(values), step)
            times = np.array(times)[indices] if hasattr(times, '__getitem__') else indices
            values = np.array(values)[indices]
            if labels is not None:
                labels = np.array(labels)[indices]
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 确定Y轴范围
        y_min, y_max = np.min(values), np.max(values)
        y_range = y_max - y_min
        y_padding = y_range * 0.1
        y_plot_min = y_min - y_padding
        y_plot_max = y_max + y_padding
        
        # 绘制背景区间
        if interval_df is not None and len(interval_df) > 0:
            # 确保interval_df有正确的列
            if 'start' in interval_df.columns and 'end' in interval_df.columns:
                for _, row in interval_df.iterrows():
                    start_idx = int(row['start'])
                    end_idx = int(row['end'])
                    
                    # 绘制异常区间背景
                    ax.axvspan(start_idx, end_idx, alpha=0.3, 
                              color=interval_color, zorder=0)
                    
                    # 添加区间标签
                    if show_interval_labels:
                        mid_idx = (start_idx + end_idx) // 2
                        ax.text(mid_idx, y_plot_max, f'[{start_idx}-{end_idx}]', 
                               ha='center', va='bottom', fontsize=7, 
                               alpha=0.7, rotation=0)
        
        # 绘制时序线
        ax.plot(times, values, color='#1f77b4', linewidth=0.8, 
                alpha=0.8, label='Time Series', zorder=2)
        
        # 标注异常点（如果有labels）
        if labels is not None:
            anomaly_mask = labels == 1
            if np.any(anomaly_mask):
                ax.scatter(times[anomaly_mask], values[anomaly_mask], 
                          c=cls.COLORS['anomaly'], s=20, zorder=5, 
                          label=f'Anomalies ({np.sum(anomaly_mask)})', alpha=0.8)
        
        # 设置坐标轴
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(y_plot_min, y_plot_max)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        # 网格
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 图例
        if labels is not None and np.any(labels == 1):
            ax.legend(loc='upper right', fontsize=10)
        
        # 添加区间统计信息
        if interval_df is not None and len(interval_df) > 0:
            n_intervals = len(interval_df)
            total_points = sum(
                int(row['end']) - int(row['start']) + 1 
                for _, row in interval_df.iterrows()
            )
            info_text = f'Intervals: {n_intervals} | Total Points: {total_points}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        
        # 保存
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    @classmethod
    def plot_precision_recall_curve(
        cls,
        precision,
        recall,
        save_path=None,
        title="Precision-Recall Curve",
        figsize=(8, 6),
        dpi=300,
        show_f1=True,
        auc_pr=None
    ):
        """
        绘制 Precision-Recall 曲线
        
        Args:
            precision: Precision 值数组
            recall: Recall 值数组
            save_path: 保存路径
            title: 图表标题
            figsize: 图表大小
            dpi: 分辨率
            show_f1: 是否显示 F1 等值线
            auc_pr: AUC-PR 值（可选）
        
        Returns:
            matplotlib.figure.Figure
        """
        cls._setup_style()
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 绘制 PR 曲线
        ax.plot(recall, precision, color='#1f77b4', linewidth=2, 
                label='PR Curve', zorder=2)
        ax.fill_between(recall, precision, alpha=0.2, color='#1f77b4')
        
        # 绘制 F1 等值线
        if show_f1:
            f1_scores = np.linspace(0.1, 0.9, 9)
            for f1 in f1_scores:
                p = f1 * recall / (2 * recall - f1 + 1e-10)
                p = np.clip(p, 0, 1)
                valid = (p > 0) & (p < 1)
                if np.any(valid):
                    ax.plot(recall[valid], p[valid], '--', color='gray', 
                           alpha=0.3, linewidth=0.5)
            
            # 标注最佳 F1 点
            f1_scores_calc = 2 * precision * recall / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores_calc)
            ax.scatter(recall[best_idx], precision[best_idx], 
                      c='red', s=100, zorder=5, marker='*',
                      label=f'Best F1={f1_scores_calc[best_idx]:.3f}')
        
        # 添加 AUC-PR 值
        if auc_pr is not None:
            ax.text(0.05, 0.95, f'AUC-PR = {auc_pr:.4f}', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 设置坐标轴
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=10)
        
        fig.tight_layout()
        
        # 保存
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    @classmethod
    def plot_score_distribution(
        cls,
        scores,
        threshold,
        labels=None,
        save_path=None,
        title="Anomaly Score Distribution",
        figsize=(10, 6),
        dpi=300,
        bins=50,
        show_stats=True
    ):
        """
        绘制异常分数分布直方图
        
        Args:
            scores: 异常分数数组
            threshold: 阈值
            labels: 真实标签（可选，用于区分正常/异常分数分布）
            save_path: 保存路径
            title: 图表标题
            figsize: 图表大小
            dpi: 分辨率
            bins: 直方图柱数
            show_stats: 是否显示统计信息
        
        Returns:
            matplotlib.figure.Figure
        """
        cls._setup_style()
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        if labels is not None:
            # 分开展示正常和异常分数分布
            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]
            
            ax.hist(normal_scores, bins=bins, alpha=0.6, color='steelblue',
                   label=f'Normal (n={len(normal_scores)})', edgecolor='white')
            ax.hist(anomaly_scores, bins=bins, alpha=0.6, color='lightcoral',
                   label=f'Anomaly (n={len(anomaly_scores)})', edgecolor='white')
        else:
            ax.hist(scores, bins=bins, alpha=0.7, color='steelblue', 
                   edgecolor='white', label=f'Samples (n={len(scores)})')
        
        # 阈值线
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold ({threshold:.3f})')
        
        # 统计信息
        if show_stats:
            stats_text = (
                f'Mean: {np.mean(scores):.3f}\n'
                f'Std: {np.std(scores):.3f}\n'
                f'Median: {np.median(scores):.3f}\n'
                f'P95: {np.percentile(scores, 95):.3f}\n'
                f'P99: {np.percentile(scores, 99):.3f}'
            )
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 设置
        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        
        # 保存
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    @classmethod
    def create_interval_df(cls, intervals: list) -> pd.DataFrame:
        """
        将区间列表转换为 DataFrame 格式
        
        Args:
            intervals: [(start, end), ...] 格式的列表
        
        Returns:
            DataFrame with columns: start, end, length
        """
        if not intervals:
            return pd.DataFrame(columns=['start', 'end', 'length'])
        
        data = []
        for start, end in intervals:
            data.append({
                'start': start,
                'end': end,
                'length': end - start + 1
            })
        
        return pd.DataFrame(data)


# 便捷函数
def plot_anomaly_timeseries(*args, **kwargs):
    """便捷函数：绘制异常时序图"""
    return DetectVisualizer.plot_anomaly_timeseries(*args, **kwargs)


def plot_anomaly_intervals(*args, **kwargs):
    """便捷函数：绘制异常区间图"""
    return DetectVisualizer.plot_anomaly_intervals(*args, **kwargs)


def plot_precision_recall_curve(*args, **kwargs):
    """便捷函数：绘制 PR 曲线"""
    return DetectVisualizer.plot_precision_recall_curve(*args, **kwargs)


def plot_score_distribution(*args, **kwargs):
    """便捷函数：绘制分数分布"""
    return DetectVisualizer.plot_score_distribution(*args, **kwargs)
