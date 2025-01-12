import torch
import torch.nn as nn
import torch.optim as optim

# 构造 XOR 数据集
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 定义多层感知机模型
class XORMLP(nn.Module):
    def __init__(self):
        super(XORMLP, self).__init__()
        self.hidden = nn.Linear(2, 4)  # 隐藏层
        self.output = nn.Linear(4, 1)  # 输出层
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x

# 初始化模型、损失函数和优化器
model = XORMLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练模型
epochs = 10000
for epoch in range(epochs):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 用户输入数据进行测试
def test_xor():
    print("\nXOR Model Ready! Enter two boolean values (0 or 1) to compute XOR.\n")
    while True:
        try:
            # 输入两个布尔值
            a = int(input("Enter first value (0 or 1): "))
            b = int(input("Enter second value (0 or 1): "))

            # 检查输入合法性
            if a not in [0, 1] or b not in [0, 1]:
                print("Invalid input! Please enter 0 or 1.")
                continue

            # 将输入转为张量
            input_data = torch.tensor([[a, b]], dtype=torch.float32)

            # 模型预测
            with torch.no_grad():
                prediction = model(input_data)
                result = (prediction > 0.5).float().item()  # 转为布尔值

            print(f"XOR({a}, {b}) = {int(result)}\n")

        except KeyboardInterrupt:
            print("\nExiting XOR Model. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}. Try again.")

# 调用测试函数
test_xor()
