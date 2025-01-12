


import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import cv2
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class ChineseCharCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChineseCharCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 64 * 64, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(-1, 128 * 64 * 64)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def predict(model, transform, class_list, frame):
    img1 = frame
    img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    img2 = transform(img1)
    img3 = img2.unsqueeze(0)

    with torch.no_grad():
        output = model(img3)
        _, predicted = torch.max(output.data, 1)
        probability = torch.softmax(output, dim=1)[0][predicted].item()
        print(f"预测结果: {class_list[predicted.item()]}, 概率: {probability}")
    return predicted.item(), probability


def test_model(model, test_loader, transform, class_list):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images[:, 0:1, :, :]  
            images = transforms.functional.resize(images, (512, 512))
            images = transforms.functional.normalize(images, (0.5,), (0.5,))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"测试集准确率: {accuracy * 100:.2f}%")


model = ChineseCharCNN(num_classes=len(os.listdir('D:\\handwritten_chinese_characters\\train')))

model.load_state_dict(torch.load('best_model.pth', weights_only=True))

model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class_list = ['北', '东', '汉', '南', '西', '字', '回','走','来','人','斯','时','验','实','可','爱','汤','沙','浩','俞','然']


test_data_dir = 'D:\\handwritten_chinese_characters\\test'
test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def paizhao3():
    cv2.namedWindow('came', 0)
    cap = cv2.VideoCapture(0)
    image_count = 0

    while True:
        ret, frame = cap.read()
        cv2.imshow('came', frame)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
        if key == 32:
            pred, prob = predict(model, transform, class_list, frame)

    cap.release()
    cv2.destroyAllWindows()


test_model(model, test_loader, transform, class_list)
paizhao3()


