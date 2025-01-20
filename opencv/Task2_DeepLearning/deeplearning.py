import torch
import torch.nn as nn
import torch.optim as optim



# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)  # 隐藏层到输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 定义训练数据
input_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
target_data = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, target_data)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# 可以开始进行用户输入获取预测结果了
while True:
    try:
        # 获取用户输入
        input_str = input("请输入两个布尔值（用空格隔开，例如 0 1，输入q退出）：")
        if input_str.lower() == "q":
            break
        input_list = list(map(int, input_str.split()))
        if len(input_list)!= 2 or any(x not in [0, 1] for x in input_list):
            print("输入格式有误，请重新输入！")
            continue

        input_tensor = torch.tensor([input_list], dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            predicted = (output > 0.5).float()
            print(f"异或运算结果为: {int(predicted.item())}")
    except:
        print("出现错误，请重新输入！")