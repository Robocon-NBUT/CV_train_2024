import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # 输入2个特征，输出4个特征
        self.fc2 = nn.Linear(4, 1)  # 输入4个特征，输出1个特征

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 第一层后接ReLU激活函数
        x = torch.sigmoid(self.fc2(x))  # 第二层后接Sigmoid激活函数
        return x

# 初始化模型、损失函数和优化器
model = XORModel()
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.1)  # 使用Adam优化器

# 训练数据
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 训练模型
for epoch in range(5000):  # 增加训练轮数
    optimizer.zero_grad()  # 清空梯度
    outputs = model(X)  # 前向传播
    loss = criterion(outputs, y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

# 定义一个函数来使用模型进行预测
def predict_xor(model, input1, input2):
    # 将输入转换为张量
    input_tensor = torch.tensor([[input1, input2]], dtype=torch.float32)
    
    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_tensor)
    
    # 将输出值四舍五入到0或1
    prediction = torch.round(output).item()
    return int(prediction)

# 获取用户输入
def get_user_input():
    while True:
        try:
            input1 = int(input("请输入第一个布尔值（0或1）："))
            input2 = int(input("请输入第二个布尔值（0或1）："))
            if input1 in [0, 1] and input2 in [0, 1]:
                return input1, input2
            else:
                print("输入无效，请输入0或1！")
        except ValueError:
            print("输入无效，请输入整数0或1！")

# 主程序
if __name__ == "__main__":
    print("欢迎使用XOR预测程序！")
    input1, input2 = get_user_input()  # 获取用户输入
    result = predict_xor(model, input1, input2)  # 使用模型进行预测
    print(f"输入: {input1}, {input2} -> 异或结果: {result}")