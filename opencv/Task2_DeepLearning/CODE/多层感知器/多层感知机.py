import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 定义一个具有两个隐藏层的 MLP 网络
        self.fc1 = nn.Linear(2, 4)  # 输入层到第一个隐藏层，输入维度为 2，输出维度为 4
        self.fc2 = nn.Linear(4, 1)  # 第一个隐藏层到输出层，输入维度为 4，输出维度为 1


    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # 使用 sigmoid 作为激活函数
        x = torch.sigmoid(self.fc2(x))  # 使用 sigmoid 作为激活函数
        return x


def main():
    # 创建一个 MLP 实例
    model = MLP()
    # 定义损失函数，使用均方误差损失
    criterion = nn.MSELoss()
    # 定义优化器，使用随机梯度下降
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 异或运算的输入和输出
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # 训练模型
    for epoch in range(10000):
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    # 测试模型
    with torch.no_grad():
        test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        predictions = model(test_inputs)
        predictions = (predictions > 0.5).float()  # 将输出转换为 0 或 1
        print(predictions)


if __name__ == "__main__":
    main()