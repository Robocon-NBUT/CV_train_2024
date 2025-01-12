import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

# XOR 数据 (输入为两布尔值)
X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_XOR = np.array([0, 1, 1, 0], dtype=np.float32).reshape(-1, 1)

# OR 数据 (输入为两布尔值)
X_OR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_OR = np.array([0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)

# 将数据转换为PyTorch张量
X_XOR_tensor = torch.tensor(X_XOR)
y_XOR_tensor = torch.tensor(y_XOR)

X_OR_tensor = torch.tensor(X_OR)
y_OR_tensor = torch.tensor(y_OR)


# 定义多层感知机（MLP）模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # 隐藏层
        x = self.fc2(x)  # 输出层
        return x


# 创建模型实例
input_size = 2  # 输入的特征数量
hidden_size = 2  # 隐藏层神经元数量
output_size = 1  # 输出层神经元数量 (0或1)

model_XOR = MLP(input_size, hidden_size, output_size)
model_OR = MLP(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数
optimizer_XOR = optim.Adam(model_XOR.parameters(), lr=0.01)
optimizer_OR = optim.Adam(model_OR.parameters(), lr=0.01)


# 训练模型的函数
def train_model(model, X_train, y_train, optimizer, epochs=1000):
    for epoch in range(epochs):
        # 前向传播
        optimizer.zero_grad()  # 清空梯度
        outputs = model(X_train)
        loss = criterion(outputs, y_train)  # 计算损失

        # 反向传播
        loss.backward()
        optimizer.step()

    return model


# 训练 XOR 模型
model_XOR = train_model(model_XOR, X_XOR_tensor, y_XOR_tensor, optimizer_XOR)

# 训练 OR 模型
model_OR = train_model(model_OR, X_OR_tensor, y_OR_tensor, optimizer_OR)

# 测试 XOR 模型
with torch.no_grad():
    y_pred_XOR = model_XOR(X_XOR_tensor)
    y_pred_XOR = torch.round(torch.sigmoid(y_pred_XOR))  # 将输出转为0或1

# 测试 OR 模型
with torch.no_grad():
    y_pred_OR = model_OR(X_OR_tensor)
    y_pred_OR = torch.round(torch.sigmoid(y_pred_OR))  # 将输出转为0或1

# 输出结果
print("XOR 预测结果：", y_pred_XOR.numpy().flatten())
print("OR 预测结果：", y_pred_OR.numpy().flatten())

# 输出准确率
print("XOR 准确率：", accuracy_score(y_XOR, y_pred_XOR.numpy()))
print("OR 准确率：", accuracy_score(y_OR, y_pred_OR.numpy()))


# 使用训练好的 XOR 模型进行实时预测
def predict_XOR(a, b):
    # 将输入转换为 PyTorch 张量
    input_tensor = torch.tensor([[a, b]], dtype=torch.float32)
    # 使用模型进行预测
    with torch.no_grad():
        output = model_XOR(input_tensor)
        prediction = torch.round(torch.sigmoid(output))  # 转换为 0 或 1
    return prediction.item()


# 用户输入两个布尔值并返回异或的计算值
def user_input_predict_XOR():
    # 获取用户输入
    a = int(input("请输入第一个布尔值 (0 或 1): "))
    b = int(input("请输入第二个布尔值 (0 或 1): "))

    # 获取并显示预测结果
    result = predict_XOR(a, b)
    print(f"{a} XOR {b} = {result}")


# 调用函数让用户输入并预测异或
user_input_predict_XOR()
