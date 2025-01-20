import torch
import torch.nn as nn
import torch.optim as optim


# 定义输入数据，XOR 运算的输入组合
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
# 定义输出数据，对应的 XOR 运算结果
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 输入层到隐藏层
        self.fc1 = nn.Linear(2, 2)
        # 隐藏层到输出层
        self.fc2 = nn.Linear(2, 1)
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# 实例化模型
model = MLP()


# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# 训练模型
for epoch in range(10000):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)


    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/10000], Loss: {loss.item():.4f}')


# 对输入数据进行预测
with torch.no_grad():
    predictions = model(X)
    rounded_predictions = torch.round(predictions)


print(rounded_predictions)