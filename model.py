
import torch.nn as nn
import torch.nn.functional as F
def Relu(x):
    return nn.functional.relu(x)
class bcinet(nn.Module):
    def __init__(self,inchannels):
        super(bcinet, self).__init__()
        self.conv = nn.Conv2d(inchannels,16,kernel_size=3,padding=1)
        self.conv1 = nn.Conv2d(16, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12,48,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(48, 16, kernel_size= 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(4 ,1024)

        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,2)
        self.bn = nn.BatchNorm2d(16,eps=1e-5,affine=True)
        self.bn1 = nn.BatchNorm2d(12, eps=1e-5, affine=True)
        self.bn2 = nn.BatchNorm2d(48,eps=1e-5,affine=True)

        self.fla = nn.Flatten()
    def forward(self,x):
        '''
        x = Relu(self.bn(self.conv(x)))
        identity = x
        x = (self.bn1(self.conv1(x)))

        x = self.conv2(x)
        x = Relu(self.conv3(x))

        x = x + identity
        x = self.pool(x)
        '''
        x = self.fla(x)

        x = Relu(self.fc1(x))
        #x = F.dropout(x,0.1)
        x = Relu(self.fc2(x))
        x = self.fc4(self.fc3(x))
        return x

