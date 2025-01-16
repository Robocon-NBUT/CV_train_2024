# 导入PyTorch库，以及其神经网络模块nn和优化器模块optim
import torch
import torch.nn as nn
import torch.optim

# 定义一个名为XOR_MLP的神经网络类，它继承自torch.nn.Module
class XOR_MLP(nn.Module):
    def __init__(self):
        # 调用父类nn.Module的构造函数
        super(XOR_MLP, self).__init__()
        # 定义第一个全连接层，输入特征数为2，输出特征数为4
        self.fc1 = nn.Linear(2, 4)
        # 定义第二个全连接层，输入特征数为4，输出特征数为1
        self.fc2 = nn.Linear(4, 1)

    # 定义网络的前向传播函数
    def forward(self, x):
        # 将输入x通过第一个全连接层，并应用tanh激活函数
        x = torch.tanh(self.fc1(x))
        # 将结果通过第二个全连接层，并应用sigmoid激活函数
        x = torch.sigmoid(self.fc2(x))
        # 返回最终的输出结果
        return x

# 定义主函数
def main():
    # 创建XOR_MLP模型的实例
    model = XOR_MLP()
    # 定义均方误差损失函数
    criterion = nn.MSELoss()
    # 定义Adam优化器，传入模型的参数和学习率0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # 定义XOR问题的输入数据，使用float32数据类型
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    # 定义XOR问题的输出数据，使用float32数据类型
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    # 进行5000次训练周期
    for epoch in range(5000):
        # 清除优化器中的梯度信息
        optimizer.zero_grad()
        # 将输入数据X传入模型，得到模型的输出结果
        output = model(X)
        # 计算模型输出结果与真实结果之间的损失值
        loss = criterion(output, y)
        # 反向传播损失值，计算梯度
        loss.backward()
        # 根据梯度更新模型的参数
        optimizer.step()
        # 每500个周期打印一次当前的周期数和损失值
        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    # 定义用于测试的输入数据
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    # 在不计算梯度的情况下进行测试
    with torch.no_grad():
        # 将测试输入数据传入模型，得到模型的预测结果
        predictions = model(test_inputs)
        # 将预测结果转换为0或1的二值结果
        predictions = (predictions > 0.5).float()
        # 打印模型的预测结果
        print(predictions)

# 如果当前脚本被直接运行，则执行主函数
if __name__ == "__main__":
    main()