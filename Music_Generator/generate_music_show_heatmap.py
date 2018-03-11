
# coding: utf-8

# In[3]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from rnn2 import rnn2,encoder
from torch.optim import lr_scheduler
import argparse


def sample_distr(distr, temperature):
    distr = distr*1.0/temperature
    distr = np.exp(distr)
    distr = distr*1.0/np.sum(distr)
    idx = np.random.choice(a.leng, 1, p=distr)
    return idx[0]

def generate_music(l, idd, T):
    model.eval()
    prime = '<start>\nX:'
    hidden = model.init_hidden()
    str_ = prime
    preds = prime
    res = np.zeros((30,20,100))
    for i in range(l):
        data_ = torch.FloatTensor(a.code(str_, False))
        data_ = Variable(data_).cuda()
        
        output, hidden = model(data_, hidden)
        aa = hidden[1].view(-1,100).cpu().data.numpy()[0]
        distr = output.view(-1,a.leng).cpu().data.numpy()[-1]
        indxx = sample_distr(distr, T)
        next_ = a.indx2char[indxx]
        for hid in range(100):
            res[int(i/20),int(i%20),int(hid)] = aa[hid]
        preds = preds+next_
        str_ = str_+next_
        if len(str_)>batch_size:
            str_ = str_[1:]
    print (preds)
    return preds,res

def show_heatmap(hid):
    plt.figure(figsize=(12,8))
    heatmap = plt.pcolor(np.tanh(res[::-1,:,hid]), cmap='coolwarm', vmin=-1., vmax=1.)
    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            char = preds[y*20+x+10]
            if char == '\n':
                char = 'nl'
            elif char == '\t':
                char = 'tl'
            elif char == ' ':
                char = 'sp'
            plt.text( x + 0.5, 30 - y - 0.5, '%s' % char,
                     horizontalalignment='center',
                     verticalalignment='center',
                     )

    plt.colorbar(heatmap)

    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqlen', default=32, type=int, help='sequence length')
    parser.add_argument('--bsz', default=32, type=int, help='batch size')
    parser.add_argument('--dropout', default=0, type=int, help='dropout rate')
    parser.add_argument('--nhidd', default=100, type=int, help='number of hidden nodes per layer')
    parser.add_argument('--id', default=0, type=int, help='id of hidden unit whose heatmap will be show')
    parser.add_argument('--inp', default='./input.txt', type=str, help='path to data file')
    parser.add_argument('--outp', default='./', type=str, help='path to model')
    args = parser.parse_args()
    
    bat = args.bsz
    batch_size = args.seqlen
    dropout = args.dropout
    nhid = args.nhidd
    inp = args.inp
    outp = args.outp
    idd = args.id
    
    if torch.cuda.is_available() == False:
        print('Cuda not found!')
    else:
        with open(inp,'r') as f:
            data = f.read()
        a = encoder(data)

        model = rnn2(n_inp=a.leng, n_hid=nhid, rnn_type='LSTM', bsz=bat, dropout=dropout).cuda()
        model.load_state_dict(torch.load(outp+'model_a.pth'))
        print(model)
        model.change_bsz(1)
        preds,res = generate_music(600, 93, T=0.5)

        show_heatmap(idd)

