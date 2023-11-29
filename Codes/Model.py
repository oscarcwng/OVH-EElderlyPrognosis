import torch.nn as nn
import torch
from torchvision import models
from . import Attention
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_ft = models.resnet101(weights='DEFAULT').to(device)
        rnmodules = list(self.model_ft.modules())
        self.conv_block1 = nn.Sequential(*rnmodules[1:6])
        self.conv_block2 = rnmodules[33]
        self.conv_block3 = rnmodules[69]
        self.conv_block4 = rnmodules[257]
        self.avgpool = rnmodules[285]
        
        self.attn1 = Attention.AttBloc(256, 512, 256, 2, normalize_attn=True)
        self.attn2 = Attention.AttBloc(1024, 2048, 256, 2, normalize_attn=True)
        
        self.fc1 = nn.Linear(3328, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)
        self.m = nn.Dropout(p=0.2)
    
    def forward(self, x):
        block1 = self.conv_block1(x)
        block2 = self.conv_block2(block1)
        block3 = self.conv_block3(block2)
        block4 = self.conv_block4(block3)
        x = self.avgpool(block4).view(x.size()[0],2048)
        a1, g1 = self.attn1(block1, block2)
        a2, g2 = self.attn2(block3, block4)
        x = torch.cat((x,g1,g2), dim=1)
        x = self.m(x)
        x = F.relu(self.fc1(x))
        x = self.m(x)
        x = F.relu(self.fc2(x))
        x = self.m(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x, a1, a2