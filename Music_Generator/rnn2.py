import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

class rnn2(nn.Module):
    def __init__(self, n_inp, n_hid=100, n_lay=1, bias=True, dropout=0, rnn_type='LSTM', bsz = 1):
        super(rnn2, self).__init__()
#         self.rnn = nn.LSTM(n_inp, hidden_dim)
        self.rnn = getattr(nn, rnn_type)(n_inp, n_hid, n_lay, dropout=dropout, bias=bias)
        self.fc = nn.Linear(n_hid, n_inp)
        self.n_hid = n_hid
        self.bsz = bsz
        self.layer = n_lay
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (Variable(torch.zeros(self.layer, self.bsz, self.n_hid)).cuda(),
                Variable(torch.zeros(self.layer, self.bsz, self.n_hid)).cuda())
    
    def forward(self, data, hidden):
        x,hidden = self.rnn(data.view(len(data),self.bsz,-1), hidden)
#         x,hidden = self.rnn(data, hidden)
        x = self.fc(x)
        return x, hidden
    
    def change_bsz(self, bsz):
        self.bsz = bsz

class encoder:
    def __init__(self, data):
        self.voc = sorted(set(data))
        self.leng = len(self.voc)
        self.char2indx = {c:i for i,c in enumerate(self.voc)}
        self.indx2char = {i:c for i,c in enumerate(self.voc)}
    
    def code(self, source, flg):
        N = len(source)
        if flg:
            y = [int(self.char2indx[source[i]]) for i in range(N)]
            return y
        y = np.zeros((N,self.leng))
        for i in range(N):
            y[i,self.char2indx[source[i]]] = 1.0
        return y