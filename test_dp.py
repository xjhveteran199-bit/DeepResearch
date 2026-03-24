import sys, pandas as pd, numpy as np
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r'C:\Users\XJH\DeepPredict\src')
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv(r'C:\Users\XJH\DeepResearch\test_data\dp_temperature.csv')
temp = df['Temp'].values.astype(np.float32)

seq_len = 30
X_seq, y_seq = [], []
for i in range(seq_len, len(temp)):
    X_seq.append(temp[i-seq_len:i].flatten())
    y_seq.append(temp[i])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print('=== DeepPredict Test (Daily Min Temperatures) ===')
print('Shape: X=%s, y=%s' % (X_seq.shape, y_seq.shape))
n = len(X_seq)

# CNN1D
from models.cnn1d_model import CNN1DPredictorV4
p1 = CNN1DPredictorV4()
ok1, msg1 = p1.train(X_seq, y_seq, seq_len=30, hidden_channels=32, num_layers=2, epochs=20, batch_size=32, learning_rate=0.001, test_size=0.1)
m1 = p1.metrics
print('CNN1D metrics: %s' % m1)

# LSTM
from models.lstm_model import LSTMPredictor
p2 = LSTMPredictor()
ok2, msg2 = p2.train(X_seq.astype(np.float32), y_seq.astype(np.float32), hidden_size=64, num_layers=2, epochs=20, batch_size=32, learning_rate=0.001, seq_len=30, test_size=0.1, target_col='Temp')
m2 = p2.metrics
print('LSTM metrics: %s' % m2)

# RandomForest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_tr, y_tr)
rf_pred = rf.predict(X_te)
r2_rf = r2_score(y_te, rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_te, rf_pred))
status_rf = 'PASS' if r2_rf > 0.3 else 'LOW'
print('RF:    R2=%.4f RMSE=%.4f - %s' % (r2_rf, rmse_rf, status_rf))

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_tr, y_tr)
gb_pred = gb.predict(X_te)
r2_gb = r2_score(y_te, gb_pred)
rmse_gb = np.sqrt(mean_squared_error(y_te, gb_pred))
status_gb = 'PASS' if r2_gb > 0.3 else 'LOW'
print('GB:    R2=%.4f RMSE=%.4f - %s' % (r2_gb, rmse_gb, status_gb))

print()
print('DeepPredict test complete.')
