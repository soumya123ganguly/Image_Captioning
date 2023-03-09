import torch
import torch.nn as nn
from torchvision import models 
from resnet50 import Resnet50FCN
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


#ToDO Fill in the __ values
class LSTMCap(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.fmodel = Resnet50FCN()
        self.embed = nn.Embedding(len(vocab), 300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=512, num_layers=2, batch_first=True)
        #self.lstm = nn.RNN(input_size=600, hidden_size=512, num_layers=2, batch_first=True, nonlinearity='relu')
        self.out = nn.Linear(512, len(vocab))
        
    def forward(self, x, yt):
        bs = len(x)
        x = self.fmodel(x).unsqueeze(1)
        e = self.embed(yt)
        xt = torch.cat((x,e), 1)
        y, _ = self.lstm(xt)
        y = self.out(y[:, :-1, :])
        y = y.permute(0, 2, 1)
        return y
    
    def gen_captions(self, pred):
        #print(pred.shape)
        pred_caption = None
        pred = pred.squeeze()
        if False:
            pred_idc = torch.argmax(pred, 1)
            pred_caption = [str(self.vocab.idx2word(pred_idx)).lower() for pred_idx in pred_idc]
        else:
            #print(pred.shape)
            pred_idc = Categorical(F.softmax(pred/0.1, dim=1)).sample()
            pred_caption = [str(self.vocab.idx2word[pred_idx.item()]).lower() for pred_idx in pred_idc]
        #return pred_caption
        return pred_idc
    
    def forward_eval(self, img):
        temperature = 0.1
        max_length = 20
        deterministic = False

        batch_size = img.shape[0]
        hidden = torch.zeros(2, 512).cuda()
        cell = torch.zeros(2, 512).cuda()
        
        text_embedded = self.fmodel(img).unsqueeze(1)
        count = True
        # TODO: opertions on text_lists is not very efficient, consider using index instead of string
        text_lists = [['<start>'] for _ in range(batch_size)]
        hidden = None
        cell = None
        while (not all([text_list[-1] == '<end>' for text_list in text_lists])) and all([len(text_list) <= max_length for text_list in text_lists]):            
            
            if not count:
                text = torch.tensor([[self.vocab(text_list[-1])]
                                 for text_list in text_lists], dtype=torch.long).to(img.device)
                text_embedded = self.embed(text)  # batch_size, seq_len, embedding_size
            if count:
                count = False
                #decode, hidden = self.lstm(text_embedded)  # seq_len, batch_size, hidden_size
                decode, (hidden, cell) = self.lstm(text_embedded)  # seq_len, batch_size, hidden_size
            else:
                #decode, hidden = self.lstm(text_embedded, hidden)
                decode, (hidden, cell) = self.lstm(text_embedded, (hidden, cell))
            out = self.out(decode).permute(0, 2, 1)  # batch_size, vocab_size, seq_len

            #if deterministic:
            #    _, text_ids = F.log_softmax(out.squeeze_(), dim=1).max(dim=1)
            #else:
            #    text_ids = Categorical(F.softmax(out.squeeze_() / temperature, dim=1)).sample()
            text_ids = self.gen_captions(out)
            for text_list, text_id in zip(text_lists, text_ids):
                if text_list[-1] != '<end>':
                    text_list.append(self.vocab.idx2word[int(text_id.item())])

        text_lists = [[text for text in text_list if text != '<pad>' and text !=
                       '<start>' and text != '<end>' and text != '<unk>'] for text_list in text_lists]

        return text_lists
