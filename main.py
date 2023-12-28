import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# 加载乳腺癌数据集
data = load_breast_cancer()

# 提取的权重和特征名称
weights = [-0.28991288, -0.28757913, -0.26442227, -0.32403195, -0.14547556, 0.33371681,
           -0.5368757, -0.6897453, 0.13738421, 0.14712086, -0.70225155, 0.168874,
           -0.42948691, -0.52564019, -0.03145542, 0.37527943, -0.10159003, -0.2145174,
           0.2191944, 0.34762611, -0.41496545, -0.67574161, -0.29618542, -0.41815282,
           -0.39832785, 0.15735567, -0.57582155, -0.37529222, -0.63764078, -0.15076565]
features = data.feature_names

# 创建特征名称和权重的映射
feature_weights = dict(zip(features, weights))

# 对权重进行排序以改善可视化
sorted_features = sorted(feature_weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
sorted_features_names, sorted_weights = zip(*sorted_features)

# 绘制条形图
plt.figure(figsize=(10, 8))
plt.barh(sorted_features_names, sorted_weights, color='skyblue')
plt.xlabel('Weight')
plt.ylabel('Feature')
plt.title('Feature Weights Visualization')
plt.gca().invert_yaxis()  # 反转y轴，以便重要特征在上方
plt.show()
