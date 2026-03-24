# -*- coding: utf-8 -*-
"""
DeepClassify 可视化模块
提供 4 个核心图表：混淆矩阵、ROC曲线、t-SNE降维、信号时序标注
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 强制禁用 LaTeX (避免找不到 latex.exe 错误)
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


# Nature 配色方案
NATURE_PALETTE = [
    '#E64B35',  # 红色
    '#4DBBD5',  # 青色
    '#00A087',  # 绿色
    '#3C5488',  # 深蓝
    '#F39B7F',  # 橙色
    '#8491B4',  # 灰蓝
    '#91D1C2',  # 浅绿
    '#DC0000',  # 深红
    '#7E6148',  # 棕色
]


# Nature 风格配置
NATURE_STYLE = {
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


class ClassifyVisualizer:
    """分类任务可视化工具类"""
    
    def __init__(self, dpi=300, figsize=(6, 5)):
        """
        初始化可视化器
        
        Args:
            dpi: 图像分辨率（默认300 DPI）
            figsize: 图像尺寸 (宽, 高)
        """
        self.dpi = dpi
        self.figsize = figsize
        self.palette = NATURE_PALETTE
        # 应用 Nature 风格
        plt.rcParams.update(NATURE_STYLE)
    
    def plot_confusion_matrix(self, y_true, y_pred, labels, save_path=None,
                             show_percent=True, show_values=True,
                             cmap='Blues', figsize=None):
        """
        绘制混淆矩阵（Nature 风格）
        
        Args:
            y_true: 真实标签（编码后 0,1,2...）
            y_pred: 预测标签
            labels: 类别名称列表
            save_path: 保存路径（可选）
            show_percent: 是否显示百分比
            show_values: 是否显示绝对数值
            cmap: 颜色映射
            figsize: 图像尺寸
        
        Returns:
            matplotlib.figure.Figure
        """
        if figsize is None:
            figsize = self.figsize
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # 准备标注文字
        annot_text = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                pct = cm_percent[i, j]
                if show_values and show_percent:
                    annot_text[i, j] = f'{val}\n({pct:.1f}%)'
                elif show_percent:
                    annot_text[i, j] = f'{pct:.1f}%'
                elif show_values:
                    annot_text[i, j] = f'{val}'
                else:
                    annot_text[i, j] = ''
        
        # 绘制热力图（纯 matplotlib）
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=15)
        
        # 设置刻度标签
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                text = annot_text[i, j]
                if text:
                    ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
        
        ax.set_xlabel('Predicted Label', fontweight='medium')
        ax.set_ylabel('True Label', fontweight='medium')
        ax.set_title('Confusion Matrix', fontweight='bold', pad=10)
        
        # 调整刻度标签角度
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_roc_curve(self, y_true, y_proba, labels, save_path=None,
                       figsize=None, show_macro=True, show_micro=True):
        """
        绘制 ROC 曲线（Nature 风格）
        
        Args:
            y_true: 真实标签（编码后 0,1,2...）
            y_proba: 预测概率矩阵 (n_samples, n_classes)
            labels: 类别名称列表
            save_path: 保存路径（可选）
            figsize: 图像尺寸
            show_macro: 是否显示 macro-average
            show_micro: 是否显示 micro-average
        
        Returns:
            matplotlib.figure.Figure
        """
        if figsize is None:
            figsize = self.figsize
        
        n_classes = y_proba.shape[1]
        
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # 存储 AUC 值
        auc_scores = {}
        
        # 颜色分配
        colors = self.palette[:n_classes]
        
        if n_classes == 2:
            # 二分类：单条曲线
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[0], linewidth=2,
                   label=f'ROC (AUC = {roc_auc:.3f})')
            auc_scores['binary'] = roc_auc
        else:
            # 多分类：One-vs-Rest
            fpr = {}
            tpr = {}
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
                roc_auc = auc(fpr[i], tpr[i])
                label = labels[i] if i < len(labels) else f'Class {i}'
                ax.plot(fpr[i], tpr[i], color=colors[i], linewidth=2,
                       label=f'{label} (AUC = {roc_auc:.3f})')
                auc_scores[label] = roc_auc
            
            # Micro-average ROC (使用 One-vs-Rest 聚合)
            if show_micro and n_classes > 2:
                # 对于多分类，micro-average 需要特殊处理
                # 计算每个样本的最大概率对应的类别
                y_true_binary = np.zeros((len(y_true), n_classes))
                for i in range(n_classes):
                    y_true_binary[:, i] = (y_true == i).astype(int)
                
                # 聚合所有类的 TPR 和 FPR
                fpr_micro = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                tpr_micro = np.zeros_like(fpr_micro)
                for i in range(n_classes):
                    tpr_micro_interp = np.interp(fpr_micro, fpr[i], tpr[i])
                    tpr_micro += tpr_micro_interp * np.sum(y_true == i)
                tpr_micro /= len(y_true)
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                
                ax.plot(fpr_micro, tpr_micro, color='navy', linewidth=2,
                       linestyle='--', label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
                auc_scores['micro'] = roc_auc_micro
            
            # Macro-average ROC
            if show_macro:
                roc_auc_macro = np.mean([auc_scores.get(labels[i], 0) for i in range(n_classes)])
                ax.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle='--',
                       label=f'Macro-average (AUC = {roc_auc_macro:.3f})')
                auc_scores['macro'] = roc_auc_macro
        
        # 对角线（随机分类器）
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='medium')
        ax.set_ylabel('True Positive Rate', fontweight='medium')
        ax.set_title('ROC Curve', fontweight='bold', pad=10)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig, auc_scores
    
    def plot_tsne(self, X_features, labels, y_true, save_path=None,
                  figsize=None, perplexity=30, n_iter=1000,
                  random_state=42, dimension=2):
        """
        绘制 t-SNE 降维可视化（Nature 风格）
        
        Args:
            X_features: 特征矩阵 (n_samples, n_features)
            y_true: 真实标签（编码后 0,1,2...）
            labels: 类别名称列表
            save_path: 保存路径（可选）
            figsize: 图像尺寸
            perplexity: t-SNE perplexity 参数
            n_iter: t-SNE 迭代次数
            random_state: 随机种子
            dimension: 2 或 3
        
        Returns:
            matplotlib.figure.Figure
        """
        if figsize is None:
            figsize = (8, 6) if dimension == 2 else (10, 8)
        
        # 确保 X 是 numpy 数组
        if hasattr(X_features, 'values'):
            X = X_features.values.astype(np.float32)
        else:
            X = np.array(X_features, dtype=np.float32)
        
        y = np.array(y_true)
        
        # t-SNE 降维
        tsne = TSNE(n_components=dimension, perplexity=perplexity,
                   max_iter=n_iter, random_state=random_state,
                   init='pca', learning_rate='auto')
        
        if dimension == 2:
            X_embedded = tsne.fit_transform(X)
        else:
            from sklearn.decomposition import PCA
            # 3D 需要先做 PCA 降维
            if X.shape[1] > 50:
                pca = PCA(n_components=50, random_state=random_state)
                X = pca.fit_transform(X)
            X_embedded = tsne.fit_transform(X)
        
        fig = None
        
        if dimension == 2:
            fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
            
            # 绘制散点图
            unique_labels = np.unique(y)
            colors = self.palette[:len(unique_labels)]
            
            for i, label in enumerate(unique_labels):
                mask = y == label
                label_name = labels[label] if label < len(labels) else f'Class {label}'
                ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                          c=colors[i], label=label_name, alpha=0.7,
                          s=50, edgecolors='white', linewidth=0.5)
            
            ax.set_xlabel('t-SNE Dimension 1', fontweight='medium')
            ax.set_ylabel('t-SNE Dimension 2', fontweight='medium')
            ax.set_title('t-SNE Visualization', fontweight='bold', pad=10)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
        else:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=figsize, dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            unique_labels = np.unique(y)
            colors = self.palette[:len(unique_labels)]
            
            for i, label in enumerate(unique_labels):
                mask = y == label
                label_name = labels[label] if label < len(labels) else f'Class {label}'
                ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], X_embedded[mask, 2],
                          c=colors[i], label=label_name, alpha=0.7, s=50)
            
            ax.set_xlabel('t-SNE 1', fontweight='medium')
            ax.set_ylabel('t-SNE 2', fontweight='medium')
            ax.set_zlabel('t-SNE 3', fontweight='medium')
            ax.set_title('t-SNE 3D Visualization', fontweight='bold', pad=10)
            ax.legend(loc='best', framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_signal_with_labels(self, signal, labels, sample_rate=1, save_path=None,
                               figsize=None, signal_name='Signal'):
        """
        绘制信号时序图 + 类别标注（适合 EMG/ECG 等传感器信号）
        
        Args:
            signal: 1D 信号数组
            labels: 与信号对应的类别标签数组（每样本一个标签）
            sample_rate: 采样率（Hz）
            save_path: 保存路径（可选）
            figsize: 图像尺寸
            signal_name: 信号名称
        
        Returns:
            matplotlib.figure.Figure
        """
        if figsize is None:
            figsize = (12, 4)
        
        signal = np.array(signal).flatten()
        labels = np.array(labels)
        
        # 时间轴
        n_samples = len(signal)
        time = np.arange(n_samples) / sample_rate
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 颜色映射
        if n_classes <= len(self.palette):
            colors = self.palette[:n_classes]
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=self.dpi,
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # 上半部分：信号波形
        ax1.plot(time, signal, 'b-', linewidth=0.8, alpha=0.8)
        ax1.set_ylabel('Amplitude', fontweight='medium')
        ax1.set_title(f'{signal_name} with Class Labels', fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([time[0], time[-1]])
        
        # 下半部分：类别标注条
        # 创建分段颜色
        segment_starts = []
        segment_ends = []
        segment_classes = []
        
        current_class = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != current_class:
                segment_starts.append(start)
                segment_ends.append(i - 1)
                segment_classes.append(current_class)
                start = i
                current_class = labels[i]
        # 最后一个段
        segment_starts.append(start)
        segment_ends.append(len(labels) - 1)
        segment_classes.append(current_class)
        
        # 绘制类别区域
        for i, (start_idx, end_idx, cls) in enumerate(zip(segment_starts, segment_ends, segment_classes)):
            start_t = time[start_idx]
            end_t = time[end_idx]
            color_idx = unique_labels.tolist().index(cls) if cls in unique_labels else cls
            ax2.axvspan(start_t, end_t, alpha=0.4, color=colors[color_idx])
        
        ax2.set_xlim([time[0], time[-1]])
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('Time (s)', fontweight='medium')
        ax2.set_ylabel('Label', fontweight='medium')
        ax2.set_yticks([])
        
        # 添加图例
        legend_patches = []
        for i, label in enumerate(unique_labels):
            label_name = label if isinstance(label, str) else f'Class {label}'
            from matplotlib.patches import Patch
            legend_patches.append(Patch(facecolor=colors[i], alpha=0.4, label=label_name))
        ax2.legend(handles=legend_patches, loc='upper right', ncol=n_classes, 
                  fontsize=8, framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig


# ============ 独立函数（便于调用）============

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None, **kwargs):
    """独立函数：绘制混淆矩阵"""
    viz = ClassifyVisualizer()
    return viz.plot_confusion_matrix(y_true, y_pred, labels, save_path, **kwargs)


def plot_roc_curve(y_true, y_proba, labels, save_path=None, **kwargs):
    """独立函数：绘制 ROC 曲线"""
    viz = ClassifyVisualizer()
    return viz.plot_roc_curve(y_true, y_proba, labels, save_path, **kwargs)


def plot_tsne(X_features, labels, y_true, save_path=None, **kwargs):
    """独立函数：绘制 t-SNE 可视化"""
    viz = ClassifyVisualizer()
    return viz.plot_tsne(X_features, labels, y_true, save_path, **kwargs)


def plot_signal_with_labels(signal, labels, sample_rate=1, save_path=None, **kwargs):
    """独立函数：绘制信号时序 + 类别标注"""
    viz = ClassifyVisualizer()
    return viz.plot_signal_with_labels(signal, labels, sample_rate, save_path, **kwargs)