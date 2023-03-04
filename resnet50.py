import torch.nn as nn
from torchvision import models 

#ToDO Fill in the __ values
class Resnet50FCN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(2048*8*8, 300)

#TODO Complete the forward pass
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.fc1(x.reshape(-1, 2048*8*8))
        return x

net=Resnet50FCN().cuda()
