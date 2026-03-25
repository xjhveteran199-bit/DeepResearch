# DeepClassify / DeepDetect 体验数据集

## DeepClassify 分类数据集

### 1. UCI Wine（已测试，推荐）
- URL: https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv
- 178样本 × 13特征，3分类
- 特点：均衡小数据集，演示效果极佳

### 2. UCI Iris（经典）
- URL: https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv
- 150样本 × 4特征，3分类（setosa/virginica/versicolor）

### 3. Breast Cancer Wisconsin
- URL: https://raw.githubusercontent.com/jbrownlee/Datasets/master/wdbc.csv
- 569样本 × 30特征，2分类（恶性/良性）
- 特点：医学数据，适合演示 Precision/Recall

### 4. Glass Identification
- URL: https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv
- 214样本 × 9特征，6分类
- 特点：痕量玻璃分类，适合演示多分类混淆矩阵

### 5. EMG Hand Gesture（适合演示 CNN1D）
- 建议自行构造或使用：
- https://www.timeseriesclassification.com/description.php?Dataset=HandOutlines

---

## DeepDetect 异常检测数据集

### 1. 模拟温度数据（直接复制到网站）
- 构造脚本：见下方 generate_anomaly_data.py

### 2. Numenta NAB 异常数据
- URL: https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv
- NYC出租车乘客数据（带已知异常事件标注）
- 特点：业界标准异常检测基准数据

### 3. 电网用电异常数据
- URL: https://raw.githubusercontent.com/jbrownlee/Datasets/master/AutoIns休斯敦.csv
- 注意：需搜索具体路径

### 4. 服务器 CPU 异常
- https://github.com/SCS-Lab/streaming_ anomaly_ detection_ benchmark/raw/master/data/cpu_utilization_correlatedCPU.csv

---

## 推荐：直接下载以下数据集到本地

```powershell
# 创建体验目录
New-Item -ItemType Directory -Force "C:\Users\XJH\DeepResearch\test_data\demo"

# DeepClassify
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv" -OutFile "C:\Users\XJH\DeepResearch\test_data\demo\wine.csv"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv" -OutFile "C:\Users\XJH\DeepResearch\test_data\demo\iris.csv"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/jbrownlee/Datasets/master/wdbc.csv" -OutFile "C:\Users\XJH\DeepResearch\test_data\demo\wdbc.csv"

# DeepDetect - NYC Taxi（异常检测标杆数据）
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv" -OutFile "C:\Users\XJH\DeepResearch\test_data\demo\nyc_taxi.csv"
```
