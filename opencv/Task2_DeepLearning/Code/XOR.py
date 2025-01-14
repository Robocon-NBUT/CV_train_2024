import torch
import torch.nn as nn
import torch.optim as optim

class MLP_XOR(nn.Module):
    def __init__(self):
        super(MLP_XOR, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型
model = MLP_XOR()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 10000
for epoch in range(epochs):
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 测试模型
while True:
    input1 = int(input("请输入第一个布尔值（0或1）："))
    input2 = int(input("请输入第二个布尔值（0或1）："))
    if input1 not in [0, 1] or input2 not in [0, 1]:
        print("输入错误，请输入0或1")
        continue
    with torch.no_grad():
        input_tensor = torch.tensor([[input1, input2]], dtype=torch.float32)
        prediction = model(input_tensor)
        prediction = (prediction > 0.5).float()
        print(f"{input1} XOR {input2} 的结果是：{prediction.item()}")