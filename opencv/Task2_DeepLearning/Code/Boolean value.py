import torch
import torch.nn as nn


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# 训练数据
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 实例化模型、定义损失函数和优化器
model = MLP()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练模型
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    predicted = model(test_X)
    predicted = (predicted > 0.5).float()
    print('Predicted values:')
    print(predicted)

# 增加用户输入部分
while True:
    try:
        input_values = input("请输入两个0或1的值，以空格分隔（输入x退出）：").split()
        if input_values[0].lower() == 'x':
            break
        if len(input_values)!= 2:
            raise ValueError("输入的数值数量不正确")
        for value in input_values:
            if value not in ['0', '1']:
                raise ValueError("输入的值必须为0或1")
        input_tensor = torch.tensor([[float(input_values[0]), float(input_values[1])]], dtype=torch.float32)
        with torch.no_grad():
            new_predicted = model(input_tensor)
            new_predicted = (new_predicted > 0.5).float()
            print(f'输入值 {input_values[0]} 和 {input_values[1]} 的异或结果是：{int(new_predicted.item())}')
    except ValueError as ve:
        print(f"输入错误：{ve}")
    except Exception as e:
        print(f"发生错误：{e}")