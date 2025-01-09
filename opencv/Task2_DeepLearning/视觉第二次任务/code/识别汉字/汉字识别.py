import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image


# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入通道 1，输出通道 32，卷积核 3x3，步长 1，填充 1
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，核 2x2，步长 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入通道 32，输出通道 64，卷积核 3x3，步长 1，填充 1
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，核 2x2，步长 2
        # 输入图像为 128x128，经过两次 2x2 的池化操作，尺寸变为 32x32
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # 全连接层，输入维度 64*32*32，输出维度 128
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 全连接层，输入维度 128，输出维度 10（假设手写汉字类别为 10 个）

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # 展开特征图
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 加载自定义数据集
trainset = ImageFolder(root=r'C:\Users\黄广松\PyCharmMiscProject\pytorch\汉字识别\train汉字', transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)


# 初始化模型、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train(model, trainloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.3f}')
                running_loss = 0.0


# 开始训练
train(model, trainloader, criterion, optimizer, epochs=10)


# 保存模型
torch.save(model.state_dict(), 'handwritten_chinese_characters_cnn.pth')


# 加载保存的模型进行预测
def load_model_and_predict():
    # 实例化模型
    model = CNN()
    # 加载保存的模型参数
    model.load_state_dict(torch.load('handwritten_chinese_characters_cnn.pth'))
    model.eval()  # 将模型设置为评估模式

    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 假设要预测的图像存储在 test_image.png 中
    test_image_path = r'C:\Users\黄广松\PyCharmMiscProject\pytorch\汉字识别\train汉字\黄\009.png'
    test_image = Image.open(test_image_path).convert('RGB')
    test_image_tensor = test_transform(test_image)
    test_image_tensor = test_image_tensor.unsqueeze(0)  # 增加一个批次维度，因为模型期望输入是 (batch_size, channels, height, width)

    # 进行预测
    with torch.no_grad():
        output = model(test_image_tensor)
        _, predicted = torch.max(output, 1)
        # 获取类别名称
        class_names = trainset.classes
        predicted_class = class_names[predicted.item()]
        print(f'预测的类别为: {predicted_class}')


# 调用加载模型和预测的函数
load_model_and_predict()