# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, 'src')
os.chdir(r'C:\Users\XJH\DeepPredict')

import numpy as np
import pandas as pd

from models.cnn1d_model import CNN1DPredictorV4

print("=== CNN1D Loss Curve Test ===")

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
y = df['Temp'].values.astype(np.float32)

p = CNN1DPredictorV4()
ok, msg = p.train(
    X=y.reshape(-1, 1), y=y,
    seq_len=30, pred_len=7,
    hidden_channels=64, num_layers=2,
    epochs=10, batch_size=32,
    target_col='Temp'
)

print("Train OK:", ok)
print("R2:", round(p.metrics.get('R2', 0), 4))
print("Losses recorded:", len(p.train_losses), "epochs")
print("Train losses:", [round(x, 4) for x in p.train_losses])
print("Val losses:", [round(x, 4) for x in p.val_losses])

# Show final plot (will block until window closed)
print("\nShowing loss curve - close window to continue...")
p.plot_loss_curve(save_path='cnn1d_loss_curve.png', show=True)
print("Done.")
