import torch
import torch.nn as nn
import torch.optim as optim

# 数据集定义
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # 输入
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)             # 输出

# 定义 MLP 模型
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()#初始化
        self.fc1 = nn.Linear(2, 4)  # 隐藏层，神经元个数可以调
        self.fc2 = nn.Linear(4, 1)  # 输出层
        self.sigmoid = nn.Sigmoid()  # 激活函数

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活
        x = self.sigmoid(self.fc2(x))  # Sigmoid 输出
        return x

# 模型、损失函数和优化器
model = XORNet()
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练过程
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


def predict_xor():
    print("\n请输入两个布尔值（0 或 1）：")
    try:
        # 获取用户输入
        x1 = int(input("输入第一个数字 (0 或 1): "))
        x2 = int(input("输入第二个数字 (0 或 1): "))

        # 检查输入是否有效
        if x1 not in [0, 1] or x2 not in [0, 1]:
            print("错误：请输入有效的布尔值（0 或 1）！")
            return

        # 将输入转换为 PyTorch 张量
        input_tensor = torch.tensor([[x1, x2]], dtype=torch.float32)

        # 进行预测
        with torch.no_grad():  # 不需要梯度计算
            output = model(input_tensor).item()  # 获取预测值
            result = round(output)  # 将概率值四舍五入为 0 或 1

        # 输出结果
        print(f"预测结果：{x1} XOR {x2} = {result}")
    except ValueError:
        print("错误：请输入有效的数字！")


# 允许用户多次输入测试
while True:
    predict_xor()
    again = input("是否继续测试？(y/n): ").strip().lower()
    if again != 'y':
        print("测试结束！")
        break
