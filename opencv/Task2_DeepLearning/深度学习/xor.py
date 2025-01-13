import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 数据集定义
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# 多层感知机模型定义
class XORMLP(nn.Module):
    def __init__(self):
        super(XORMLP, self).__init__()
        self.hidden = nn.Linear(2, 4)  # 隐藏层：输入2个节点，输出4个节点
        self.output = nn.Linear(4, 1)  # 输出层：输入4个节点，输出1个节点
        self.activation = nn.Sigmoid()  # 激活函数

    def forward(self, x):
        x = self.activation(self.hidden(x))  # 隐藏层+激活函数
        x = self.activation(self.output(x))  # 输出层+激活函数
        return x


# 模型实例化
model = XORMLP()

# 损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降优化器

# 模型训练
epochs = 10000
for epoch in range(epochs):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_output = model(X)
    predictions = (test_output > 0.5).float()
    print("\n输入: \n", X.numpy())
    print("预测结果: \n", predictions.numpy())
