import torch.nn as nn
from torchvision import models 

#ToDO Fill in the __ values
class Resnet50FCN(nn.Module):

    def __init__(self, feature_size):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, feature_size)

    def forward(self, x):
        x = self.model(x)
        return x