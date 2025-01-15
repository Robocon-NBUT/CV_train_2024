import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import cv2


# 构建卷积神经网络模型类
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 假设图像预处理后为32x32，经过两次池化后变为8x8
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# 自定义数据集类
class HandwrittenChineseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(data_dir))
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# 训练模型
def train_model(model, train_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
    return model



if __name__ == "__main__":
    # 定义相关参数
    image_size = 32
    num_classes = 14  # 假设手写汉字有20个类别，可根据实际数据库中的类别数量调整
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 120
    train_data_dir = 'D:/data2/train'  # 修改为你实际存放训练数据的总目录，需按类别分文件夹存放图像

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = HandwrittenChineseDataset(train_data_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型实例并训练
    model = CNN(num_classes)
    trained_model = train_model(model, train_loader, num_epochs, learning_rate)

    # 使用摄像头进行识别
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Handwritten Chinese Character', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):  # 按下'c'键捕获图像
                captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                captured_image = cv2.resize(captured_image, (image_size, image_size))
                captured_image = torch.from_numpy(captured_image).float().unsqueeze(0).unsqueeze(0)
                captured_image = captured_image / 255.0

                with torch.no_grad():
                    output = trained_model(captured_image)
                    _, predicted_idx = torch.max(output, 1)
                    predicted_class = train_dataset.class_names[predicted_idx.item()]
                print(f"The recognized Chinese character is: {predicted_class}")
                break
        else:
            print("无法获取摄像头画面，请检查摄像头是否正常连接。")
            break
    cap.release()
    cv2.destroyAllWindows()