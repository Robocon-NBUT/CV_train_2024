import torch
import torch.nn as nn
from torchvision.transforms import transforms, RandomRotation, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
import os
import cv2 as cv
from PIL import Image
import numpy as np
def imshow(name,image):
    cv.imshow(name,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

device=torch.device("cuda"if torch.cuda.is_available() else "cpu")

#图片预处理
transform=transforms.Compose([
    transforms.Resize((512,512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])

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
model_path='myself/model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
else:
    print("找不到文件")
chinese=['东','南','西','北','汉','喆','陈','浩','沙','汤','爱','可','实','验','时','斯','人','来','走','回']

image_path='C:/Users/20936/Desktop/AI/hanzi/test'
for file_name in os.listdir(image_path):
    file_full_path = os.path.join(image_path, file_name)
    try:
        pil_image = Image.open(file_full_path).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output.data, 1)
            predicted_index = predicted.item()
            if predicted_index < len(chinese):
                print(f"图像 {file_name} 的预测结果为: {chinese[predicted_index]}")
                imshow('chinese',cv_image)
            else:
                print(f"图像 {file_name} 的预测结果索引超出范围")
    except Exception as e:
        print(f"处理图像 {file_name} 时出错: {e}")