import torch
import torch.nn as nn
import torch.optim as optim

# 数据准备
inputs = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=torch.float32)

targets = torch.tensor([
    [0, 1],
    [1, 1],
    [1, 1],
    [0, 1]
], dtype=torch.float32)


# 模型定义
class XOR_MLP(nn.Module):
    def __init__(self):
        super(XOR_MLP, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.output = nn.Linear(4, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


# 初始化
model = XOR_MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 训练
num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试
with torch.no_grad():
    predictions = model(inputs)
    predicted = (predictions > 0.5).float()
    print("\n测试结果:")
    print("输入:\n", inputs)
    print("真实输出:\n", targets)
    print("模型预测:\n", predicted)
