import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class XORModel(nn.Module):
     def __init__(self):

         super(XORModel,self).__init__()
         self.fc1=nn.Linear(2,4)
         self.fc2=nn.Linear(4,1)

     def forward(self,x):
         x=F.relu(self.fc1(x))
         x=torch.sigmoid(self.fc2(x))

         return x

X=torch.tensor([[0.0,0.0],
               [0.0,1.0],
               [1.0,0.0],
               [1.0,1.0]],dtype=torch.float32)
Y=torch.tensor([[0.0],[1.0],[1.0],[0.0]],dtype=torch.float32)

model=XORModel()

criterion=nn.MSELoss()

optimizer=optim.Adam(model.parameters(),lr=0.1)

epochs=10000
for epoch in range(epochs):
    outputs=model(X)
    loss=criterion(outputs,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch+1)%1000==0:
        print(f'Epoch[{epoch+1}/{epochs}],Loss；{loss.item():4f}')


def predict(x_input):
    x_tensor=torch.tensor(x_input,dtype=torch.float32)
    with torch.no_grad():
        predictions = model(x_tensor)
        predicted_class = (predictions > 0.5).float()
        return predicted_class


while True:
    try:
        a=float(input("请输入第一个布尔值（0或1）："))
        b=float(input("请输入第二个布尔值（0或1）："))

        result=predict([a,b])
        print(f"XOR({a},{b})={result.item()}")

    except ValueError:
        print("无效")
    except KeyboardInterrupt:
        print("\n程序已经被用户中断")
        break