import torch
x=torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float)
y=torch.tensor([[0],[1],[1],[0]],dtype=torch.float)

class Sj(torch.nn.Module):
    def __init__(self):
        super(Sj,self).__init__()
        self.cen1=torch.nn.Linear(2,2)
        self.cen2=torch.nn.Linear(2,3)
        self.cen3=torch.nn.Linear(3,1)

    def forward(self,x):
        x=torch.sigmoid(self.cen1(x))
        x=torch.sigmoid(self.cen2(x))
        x=torch.sigmoid(self.cen3(x))
        return x

moxing=Sj()

loss=torch.nn.MSELoss()
tidu = torch.optim.SGD(moxing.parameters(), lr=0.1)

for i in range(100000):
    y_hat=moxing(x)
    lo=loss(y_hat,y)
    tidu.zero_grad()
    lo.backward()
    tidu.step()
    if(i%1000==0):
        print("第",i,"轮 loss:",lo.item())

moxing.eval()
with torch.no_grad():
    y_h=moxing(x)
    yk=(y_h > 0.5).float()

for u in range(len(x)):
    print(f"输入: {x[u].numpy()}, 预测输出: {yk[u].item()}")