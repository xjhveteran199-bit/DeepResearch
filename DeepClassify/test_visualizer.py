# -*- coding: utf-8 -*-
"""Test the visualizer module"""
import sys
sys.path.insert(0, 'src')
from visualizer import ClassifyVisualizer
import numpy as np

# Quick test
viz = ClassifyVisualizer(dpi=150)

# Test confusion matrix
y_true = np.array([0,1,2,0,1,2,0,0,1,2])
y_pred = np.array([0,1,1,0,2,2,0,1,1,2])
labels = ['A','B','C']

fig = viz.plot_confusion_matrix(y_true, y_pred, labels, save_path=None)
print('Confusion Matrix: OK')

# Test ROC curve
y_proba = np.random.rand(10, 3)
y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
fig, aucs = viz.plot_roc_curve(y_true, y_proba, labels)
print('ROC Curve: OK, AUCs:', aucs)

# Test t-SNE
X = np.random.rand(30, 4)
fig = viz.plot_tsne(X, labels, y_true, perplexity=10, dimension=2)
print('t-SNE 2D: OK')

# Test signal with labels
signal = np.sin(np.linspace(0, 10, 200))
sig_labels = np.array([0]*50 + [1]*50 + [2]*50 + [0]*50)
fig = viz.plot_signal_with_labels(signal, sig_labels, sample_rate=10)
print('Signal with Labels: OK')

print('All tests passed!')