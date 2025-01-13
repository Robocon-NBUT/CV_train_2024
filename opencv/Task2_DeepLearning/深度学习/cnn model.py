import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import rcParams
import tkinter as tk
from tkinter import messagebox
import random

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载数据集路径
dataset_path = r'C:\Users\ORIENTCG\PycharmProjects\PythonProject1\dataset'
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

# 加载训练数据
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# 加载测试数据
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 获取类别名称和数量
CHARACTERS = train_dataset.classes
num_classes = len(CHARACTERS)
print(f"Number of classes: {num_classes}")


# 定义CNN神经网络模型
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64 * 64 * 64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# 模型训练和测试
def load_model():
    model = CNN(num_classes=num_classes).to(device)
    model_path = 'model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("已加载现有模型。")
    else:
        # 训练模型
        model.train()
        num_epochs = 10
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] 完成。平均损失: {epoch_loss:.4f}")

        torch.save(model.state_dict(), model_path)
        print("训练完成，模型已保存。")
    return model


# 随机抽取并显示图像及预测结果
def show_random_image():
    model.eval()
    # 随机抽取一批数据
    data_iter = iter(test_loader)
    inputs, labels = next(data_iter)

    # 随机选择一张图片
    rand_idx = random.randint(0, inputs.size(0) - 1)
    input_img = inputs[rand_idx].unsqueeze(0).to(device)
    true_label = CHARACTERS[labels[rand_idx].item()]

    # 预测
    with torch.no_grad():
        output = model(input_img)
        _, predicted = torch.max(output.data, 1)
        pred_label = CHARACTERS[predicted.item()]

    img = input_img.cpu().numpy().squeeze()
    img = (img * 0.5) + 0.5  # 反标准化
    plt.imshow(img, cmap='gray')
    plt.title(f"真实: {true_label}, 预测: {pred_label}")
    plt.axis('off')
    plt.show()


# 创建UI
def create_ui():
    window = tk.Tk()
    window.title("CNN模型测试")

    load_button = tk.Button(window, text="加载并训练模型", command=lambda: load_model())
    load_button.pack(pady=20)

    test_button = tk.Button(window, text="随机抽取并显示图像", command=show_random_image)
    test_button.pack(pady=20)

    exit_button = tk.Button(window, text="退出", command=window.quit)
    exit_button.pack(pady=20)

    window.mainloop()


if __name__ == '__main__':
    model = load_model()
    create_ui()
