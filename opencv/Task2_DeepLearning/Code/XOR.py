import torch
import torch.nn as nn
import torch.optim as optim
a=int(input())
b=int(input())

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 8)  # 进一步增加隐藏层神经元数量
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        return out


# 输入数据
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
# 标签数据
y = torch.tensor([[0.], [1.], [1.], [0.]])

model = MLP()

# 使用Xavier初始化权重
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器并调整学习率

# 增加训练轮数
for epoch in range(100000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10000 == 0:
        print(f'Epoch [{epoch + 1}/100000], Loss: {loss.item():.4f}')


def xor_predict(a, b):
    input_tensor = torch.tensor([[a, b]], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        result = (output > 0.5).float().item()
    return bool(result)


# 测试自定义函数
print(xor_predict(a, b))
