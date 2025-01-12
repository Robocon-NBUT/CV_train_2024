import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
x=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y=torch.tensor([[0.],[1.],[1.],[0.]])
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(2,8)
        self.fc2=nn.Linear(8,1)
        self.activation=nn.Sigmoid()
    def forward(self,x):
        x=self.activation(self.fc1(x))
        x=self.fc2(x)
        return x 
model=MLP()
criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)
if os.path.exists(r'C:\Users\20936\Desktop\AI\net\perceptron.pth')==False:
    for epoch in range(2000):
        outputs=model(x)
        loss=criterion(outputs,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch +1) % 100==0:
            print(f'Epoch{epoch+1}/1000,Loss:{loss.item()}')
        if (epoch+1)%1000==0:
            torch.save(model,r'C:\Users\20936\Desktop\AI\net\perceptron.pth')
net = torch.load(r'C:\Users\20936\Desktop\AI\net\perceptron.pth', weights_only=False)
net.eval()
with torch.no_grad():
    while True:
        data=input("数据为浮点型（用空格分隔每个数值）输入要预测的数据（输入 'q' 退出）：")
        if data=='q':
            break
        data_list=list(map(float,data.split()))
        datatensor=torch.tensor([data_list]).float()
        predictions=net(datatensor)
        if predictions.item()<0.5:
            print("预测结果：",0)
        else:
             print("预测结果：",1)