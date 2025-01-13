import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0006
IMAGE_SIZE = 64
CHARACTERS = ['一', '七', '三', '上', '九', '二', '五', '以', '八', '六', '十', '厂', '古', '可', '四', '大', '小', '左', '牛', '羊']

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=15, scale=(0.8, 1.2)),
    transforms.ToTensor()
])

# 加载数据集
dataset_path = r'C:\Users\许奇峰\PycharmProjects\PythonProject\dataset'
train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=transform)

# 数据集划分
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 256),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = CNN(num_classes=len(CHARACTERS)).to(DEVICE)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# 训练模型
def train():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss, val_accuracy = evaluate(val_loader)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        scheduler.step()



# 评估模型
def evaluate(loader):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(loader), 100 * correct / total

# 测试模型
def test():
    test_loss, test_accuracy = evaluate(test_loader)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# 绘制混淆矩阵
def plot_confusion_matrix(loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CHARACTERS)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.show()

# 实时识别
def live_recognition():
    model.eval()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        pil_image = Image.fromarray(resized)
        tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

        # 调试打印输入张量
        print(tensor.shape, tensor.min().item(), tensor.max().item())

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            label = CHARACTERS[predicted.item()]
            print(f"Predicted label: {label}")

        cv2.putText(frame, f'Recognized: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 主函数
if __name__ == '__main__':
    if not os.path.exists('hanzi_model.pth'):
        print("No model weights found. Training a new model.")
        train()
        torch.save(model.state_dict(), 'hanzi_model.pth')
    else:
        try:
            model.load_state_dict(torch.load('hanzi_model.pth', map_location=DEVICE))
        except RuntimeError as e:
            print(f"Model loading failed: {e}")
            print("Re-training the model with the updated structure.")
            train()
            torch.save(model.state_dict(), 'hanzi_model.pth')

    test()
    plot_confusion_matrix(test_loader)
    live_recognition()
