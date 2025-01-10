import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomRotation(10),
    transforms.RandomCrop((28, 28)),
    transforms.RandomHorizontalFlip()
])

# 加载训练集和测试集，需提前准备好数据存放路径
train_dataset = datasets.ImageFolder(root=r'C:\Users\Elmo\source\repos\SHIBie chinese\train_data', transform=transform)
test_dataset = datasets.ImageFolder(root=r'C:\Users\Elmo\source\repos\SHIBie chinese\test_data', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义卷积神经网络模型
class HandwritingCNN(nn.Module):
    def __init__(self):
        super(HandwritingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # 使用自适应池化，确保输出固定大小
        self.adaptive_pool = nn.AdaptiveMaxPool2d((5, 5))

        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, len(train_dataset.classes))

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = HandwritingCNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

# 记录损失
losses = []

# 训练模型
for epoch in tqdm(range(10)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    scheduler.step()
    losses.append(running_loss / len(train_loader))
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 绘制损失曲线
plt.plot(range(10), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')

# 保存模型
torch.save(model.state_dict(), 'handwritten_chinese_model.pth')