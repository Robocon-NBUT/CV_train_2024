import torch
import torch.nn as nn
import torch.optim as optim


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# 准备数据
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 初始化模型、损失函数和优化器
model = MLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_outputs = model(X)
    predicted = (test_outputs > 0.5).float()
    print('Predicted:', predicted)
    accuracy = (predicted == y).float().mean()
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')

# 新输入数据预测
new_input = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
with torch.no_grad():
    new_outputs = model(new_input)
    new_predicted = (new_outputs > 0.5).float()
    print('New Input Prediction:', new_predicted)