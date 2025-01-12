import multiprocessing
import torch
import cv2 as cv
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, RandomRotation, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms.functional import to_tensor, normalize
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision.io import read_image

#device=GPU
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")

#图片预处理
transform=transforms.Compose([
    transforms.RandomRotation(degrees=(-30,30)),
    transforms.Resize((512,512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])
#加载图片集
image_path='C:/Users/20936/Desktop/AI/hanzi'
dataset=ImageFolder(root=image_path,transform=transform)
dataLoader=DataLoader(dataset,batch_size=64,shuffle=True)

#划分验证集
total_size=len(dataset)
valid_size=int(total_size*0.1)
train_size=total_size-valid_size
train_dataset,valid_dataset=random_split(dataset,[train_size,valid_size])

#训练集和验证集的dataloader
train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
valid_dataloader=DataLoader(valid_dataset,batch_size=64,shuffle=False)

#CNN神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,16,kernel_size=5,stride=1,padding=2)
        self.relu1=nn.ReLU()
        self.poll1=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.poll2=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.relu3=nn.ReLU()
        self.poll3=nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc=nn.Linear(64*64*64,20)
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.poll1(x)

        x=self.conv2(x)
        x=self.relu2(x)
        x=self.poll2(x)

        x=self.conv3(x)
        x=self.relu3(x)
        x=self.poll3(x)

        x=x.view(x.size(0),-1)
        x=self.fc(x)

        return x
model=CNN()
model.to(device)
print(model)

#损失函数
criterion=nn.CrossEntropyLoss()

#优化器
optimizer=optim.Adam(model.parameters(),lr=0.0006)

#训练次数
epoches=35

#存放损失率及正确率
train_losses = []
valid_losses = []
valid_accuracies = []

for epoch in range(epoches):
    model.train()
    running_train_loss=0.0
    for inputs,labels in train_dataloader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_train_loss+=loss.item()*inputs.size(0)
    epoch_train_loss=running_train_loss/len(train_dataset)
    train_losses.append(epoch)

    model.eval
    running_valid_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for inputs,labels in valid_dataloader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_valid_loss += loss.item() * inputs.size(0)

            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    epoch_valid_loss=running_valid_loss/len(valid_dataset)
    valid_losses.append(epoch_valid_loss)
    epoch_valid_accuracy=correct/total
    valid_accuracies.append(epoch_valid_accuracy)
    print(f'Epoch {epoch + 1}/{epoches}, Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}, Valid Accuracy: {epoch_valid_accuracy}')
plt.plot(range(1, epoches + 1), train_losses, label='Train Loss')
plt.plot(range(1, epoches + 1), valid_losses, label='Valid Loss')
plt.plot(range(1, epoches + 1), valid_accuracies, label='Valid Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training and Validation Metrics')
plt.legend()
plt.show()
torch.save(model.state_dict(),'myself/model.pth')