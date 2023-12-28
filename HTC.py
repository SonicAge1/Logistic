from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# 加载乳腺癌数据集
data = load_breast_cancer()
X, y = data.data, data.target

# 数据分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 初始化SGDClassifier，使用hinge损失函数，它等同于一个线性SVM
sgd_clf = SGDClassifier(loss='hinge', max_iter=1, tol=None, learning_rate='constant', eta0=0.001, warm_start=True, random_state=42)

# 记录训练和验证的loss（这里我们用hinge loss的代理）
train_losses = []
val_losses = []


# 由于hinge loss并不提供概率输出，我们用决策函数的绝对值作为loss的代理
def proxy_hinge_loss(decision_function, y):
    # hinge loss for binary classification 1, -1 labels
    return np.mean(np.maximum(1 - y * decision_function, 0))


# 训练模型并记录代理loss
for epoch in range(200):
    sgd_clf.fit(X_train, y_train)
    decision_function_train = sgd_clf.decision_function(X_train)
    decision_function_val = sgd_clf.decision_function(X_val)

    # 使用决策函数的绝对值作为代理loss
    train_loss = proxy_hinge_loss(decision_function_train, 2 * y_train - 1)  # 将标签转换为1和-1
    val_loss = proxy_hinge_loss(decision_function_val, 2 * y_val - 1)  # 将标签转换为1和-1

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/200], Training Proxy Loss: {train_loss:.4f}, Validation Proxy Loss: {val_loss:.4f}')

# 硬阈值分类器没有predict_proba方法，所以我们使用sign函数和决策函数
y_pred = np.sign(sgd_clf.decision_function(X_val))
y_pred = (y_pred + 1) // 2  # 将-1, 1标签转换回0, 1标签

# 计算准确度
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_val, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# 绘制训练和验证代理Loss曲线
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Proxy Loss')
plt.plot(val_losses, label='Validation Proxy Loss')
plt.xlabel('Epoch')
plt.ylabel('Proxy Loss')
plt.title('Proxy Loss Curve during Training')
plt.legend()
plt.show()

# 输出学习到的参数 w 和 b
weights = sgd_clf.coef_[0]
intercept = sgd_clf.intercept_[0]
print(f"Weights: {weights}")
print(f"Intercept: {intercept}")

