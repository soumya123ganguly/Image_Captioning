import torch
import torch.nn as nn
from torchvision import models 
from resnet50 import Resnet50FCN

#ToDO Fill in the __ values
class LSTMCap(nn.Module):

    def __init__(self, len_vocab):
        super().__init__()
        self.fmodel = Resnet50FCN()
        self.embed = nn.Embedding(len_vocab, 300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=512, num_layers=2, batch_first=True)
        self.out = nn.Linear(512, len_vocab)

#TODO Complete the forward pass
    def forward(self, x, yt):
        bs = len(x)
        x = self.fmodel(x).unsqueeze(1)
        e = self.embed(yt)
        xt = torch.cat((x,e), 1)
        h0 = torch.zeros(2, bs, 512).cuda()
        c0 = torch.zeros(2, bs, 512).cuda()
        y, _ = self.lstm(e, (h0, c0))
        y = self.out(y)
        y = y.permute(0, 2, 1)
        return y
