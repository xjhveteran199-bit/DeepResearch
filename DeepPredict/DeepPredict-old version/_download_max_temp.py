import urllib.request, os, sys
sys.path.insert(0, '.')

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-max-temperatures.csv'
dest = r'data\daily_max_temperatures.csv'
urllib.request.urlretrieve(url, dest)
print('Downloaded:', dest)

import pandas as pd
df = pd.read_csv(dest)
print(f'Shape: {df.shape}')
print(df.head(3))
print(f'Columns: {df.columns.tolist()}')
