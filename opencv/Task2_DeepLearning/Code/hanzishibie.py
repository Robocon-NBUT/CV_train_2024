import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),       # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])


# 加载数据集
def load_dataset(data_dir):
    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return train_loader, train_dataset


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, len(train_dataset.classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = out.view(-1, 32 * 16 * 16)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out


# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')


# 保存模型
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


# 加载模型
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


# 预测
def predict(model, test_images_dir, train_dataset):
    for root, dirs, files in os.walk(test_images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_image_path = os.path.join(root, file)
                test_image = datasets.folder.default_loader(test_image_path)
                test_image = transform(test_image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(test_image)
                    _, predicted = torch.max(outputs.data, 1)
                    print(f'图像 {test_image_path} 的预测结果: {train_dataset.classes[predicted.item()]}')


# 评估模型准确率
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    data_dir = r"C:\Users\35453\Desktop\hanzi\train"
    model_path = 'cnn_hanzi.pth'
    test_images_dir = r"C:\Users\35453\Desktop\hanzi\test"

    train_loader, train_dataset = load_dataset(data_dir)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    save_model(model, model_path)

    model = load_model(model, model_path)

    # 加载测试集
    test_dataset = datasets.ImageFolder(root=test_images_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    accuracy = evaluate_model(model, test_loader)
    print(f'模型在测试集上的准确率: {accuracy}')

    predict(model, test_images_dir, train_dataset)
