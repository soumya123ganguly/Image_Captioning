import torch
import torch.nn as nn
from torchvision import models 
from resnet50 import Resnet50FCN
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

#ToDO Fill in the __ values
class CaptionsLSTM1(nn.Module):

    def __init__(self, hidden_size, embedding_size, vocab, captions):
        super().__init__()
        
        # Hyperparameters
        self.vocab = vocab
        self.captions = captions
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        # Model
        self.resnet50 = Resnet50FCN(embedding_size)
        self.embed = nn.Embedding(len(vocab), embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_size, len(vocab))
        
    def forward(self, x, yt):
        x = self.resnet50(x).unsqueeze(1)
        e = self.embed(yt)
        xt = torch.cat((x,e), 1)
        y, _ = self.lstm(xt)
        y = self.out(y[:, :-1, :])
        y = y.permute(0, 2, 1)
        return y
    
    def gen_captions(self, pred):
        pred_caption = None
        if self.captions['deterministic']:
            pred_idc = torch.argmax(pred, 1)
            pred_caption = [str(self.vocab.idx2word(pred_idx)).lower() for pred_idx in pred_idc]
        else:
            pred_idc = Categorical(F.softmax(pred/self.captions['temperature'], dim=1)).sample()
            pred_caption = [str(self.vocab.idx2word[pred_idx.item()]).lower() for pred_idx in pred_idc]
        return pred_caption

    def gen_text_captions(self, img):
        feature_vec = self.resnet50(img).unsqueeze(1)
        pred, (hn, cn) = self.lstm(feature_vec)
        pred = self.out(pred).squeeze()
        pred_captions = []
        last_caption = self.gen_captions(pred)
        for cap in last_caption:
            pred_captions.append([cap])
        for _ in range(self.captions['max_length']):
            if all([pred_caption[-1] == '<end>' for pred_caption in pred_captions]):
                break
            ohe_text = torch.tensor([[self.vocab.word2idx[pred_caption[-1]]]for pred_caption in pred_captions]).to(img.device)
            embed_text = self.embed(ohe_text)
            pred, (hn, cn) = self.lstm(embed_text, (hn, cn))
            pred = self.out(pred).squeeze()
            last_caption = self.gen_captions(pred)
            for i, pred_cap in enumerate(pred_captions):
                if pred_cap[-1] != '<end>':
                    pred_cap.append(last_caption[i])
        fix_pred_captions = []
        for pred_cap in pred_captions:
            fix_pred_cap = []
            for cap in pred_cap:
                if cap not in ['<start>', '<end>', '<pad>', '<unk>']:
                    fix_pred_cap.append(cap)
            fix_pred_captions.append(fix_pred_cap)
        return fix_pred_captions