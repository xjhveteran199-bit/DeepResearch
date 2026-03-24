# -*- coding: utf-8 -*-
"""Test the visualizer module"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

import sys
sys.path.insert(0, 'src')
from visualizer import ClassifyVisualizer
import numpy as np

# Quick test
viz = ClassifyVisualizer(dpi=150)

# Test 1: Confusion matrix (multiclass)
y_true = np.array([0,1,2,0,1,2,0,0,1,2])
y_pred = np.array([0,1,1,0,2,2,0,1,1,2])
labels = ['A','B','C']

fig = viz.plot_confusion_matrix(y_true, y_pred, labels, save_path=None)
print('Confusion Matrix (multiclass): OK')

# Test 2: ROC curve (binary)
y_true_binary = np.array([0,0,0,0,0,1,1,1,1,1])
y_proba_binary = np.random.rand(10, 2)
y_proba_binary = y_proba_binary / y_proba_binary.sum(axis=1, keepdims=True)
labels_binary = ['Neg', 'Pos']

fig, aucs = viz.plot_roc_curve(y_true_binary, y_proba_binary, labels_binary)
print('ROC Curve (binary): OK, AUCs:', aucs)

# Test 3: ROC curve (multiclass - OvR)
y_true_multi = np.array([0,1,2,0,1,2,0,0,1,2])
y_proba_multi = np.random.rand(10, 3)
y_proba_multi = y_proba_multi / y_proba_multi.sum(axis=1, keepdims=True)
labels_multi = ['A','B','C']

fig, aucs = viz.plot_roc_curve(y_true_multi, y_proba_multi, labels_multi)
print('ROC Curve (multiclass): OK, AUCs:', aucs)

# Test 4: t-SNE
X = np.random.rand(30, 4)
y_tsne = np.array([0]*10 + [1]*10 + [2]*10)
labels_tsne = ['A','B','C']
fig = viz.plot_tsne(X, labels_tsne, y_tsne, perplexity=10, dimension=2)
print('t-SNE 2D: OK')

# Test 5: Signal with labels
signal = np.sin(np.linspace(0, 10, 200))
sig_labels = np.array([0]*50 + [1]*50 + [2]*50 + [0]*50)
fig = viz.plot_signal_with_labels(signal, sig_labels, sample_rate=10)
print('Signal with Labels: OK')

print('\n=== All tests passed! ===')