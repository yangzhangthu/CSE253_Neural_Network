
# coding: utf-8

# In[4]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from rnn2 import rnn2, encoder
from torch.optim import lr_scheduler
import argparse




def get_batch(data, seq_leng, is_random=True, start_pos=0):
    if is_random:
        leng = len(data)
        start_pos = np.random.randint(0, leng-seq_leng, size=1)[0]
        xdata = data[start_pos: start_pos+seq_leng]
        label = data[start_pos+1: start_pos+seq_leng+1]
        return xdata, label, start_pos
    else:
        leng = len(data)
        if start_pos+seq_leng+1 > leng:
            start_pos = leng-seq_leng-1
        xdata = data[start_pos: start_pos+seq_leng]
        label = data[start_pos+1: start_pos+seq_leng+1]
        return xdata, label, start_pos+seq_leng

def get_batch_(data, seq_leng, is_random=False, start_pos=0):
    xdata = np.zeros((seq_leng, a.leng, bat))
    label = []
    
    for batid in range(bat):
        data_, target_, start_ = get_batch(data[batid], seq_leng, False, start_pos)
        xdata[:,:,batid] = a.code(data_, False)
#         label.extend(a.code(target_, True))
        label.append(a.code(target_, True))

    label = np.array(label,dtype=int)

    label = np.transpose(label).reshape(1, bat*batch_size).tolist()[0]
  
    return xdata, label, start_pos+seq_leng

def eva(dataset, debug=True, is_random=True):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden()
    itrn = int(len(dataset[0])/batch_size)
    start_ = 0
    if debug:
        itrn = 10
    for i in range(itrn):
        hidden = tuple(Variable(v.data).cuda() for v in hidden)
        model.zero_grad()
        
        batch_, batar_, start_ = get_batch_(dataset, batch_size, is_random=False, start_pos=start_)
        
        data_ = torch.transpose(torch.FloatTensor(batch_),1,2)
        target_ = torch.LongTensor(batar_)
        
        data_, target_ = Variable(data_).cuda(), Variable(target_).cuda()
        
        output, hidden = model(data_, hidden)

        loss = criterion(output.view(-1, a.leng), target_)

        total_loss += loss.data
        
    return total_loss*1.0/itrn

def batchify(dataset):
    batch_leng = int(np.floor(len(dataset)*1.0/bat))
    new_data = []
    start = 0
    for i in range(bat):
        new_data.append(dataset[start:start+batch_leng])
        start = start + batch_leng
    return new_data

def plot_loss():
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.grid()
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train','validation'],loc=1)
	plt.show()
    

def train(iter_n, is_random=True):
    
    trn = int(len(tr_data_[0])/batch_size)
    for epoch in range(iter_n):
        start_time = time.time()
        total_loss = 0
        start_ = 0
        model.train()
        hidden = model.init_hidden()
        for i in range(trn):
#         for i in range(5):
            hidden = tuple(Variable(v.data).cuda() for v in hidden)
            model.zero_grad()

            batch_, batar_, start_ = get_batch_(tr_data_, batch_size, is_random=False, start_pos=start_)
#             print batar_
            data_ = torch.transpose(torch.FloatTensor(batch_),1,2)
            target_ = torch.LongTensor(batar_)
            data_, target_ = Variable(data_).cuda(), Variable(target_).cuda()

            output, hidden = model(data_, hidden)   
#             print(output.view(-1, a.leng).data.shape)
#             print (target_)
            loss = criterion(output.view(-1, a.leng), target_)
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            if i % trn == trn-1:
                print('[%d %d] loss: %f' % (epoch, i+1, total_loss.cpu().numpy()[0]/trn))
                total_loss = 0
                print(time.time()-start_time)
                start_time = time.time()
        start_time = time.time()
        train_loss.append(eva(tr_data_, debug=False, is_random=is_random).cpu().numpy()[0])
        print('Loss on training set: %f' % (train_loss[-1]) )
        print(time.time()-start_time)
        start_time = time.time()
        val_loss.append(eva(val_data_, debug=False, is_random=is_random).cpu().numpy()[0])
        print('Loss on validation set: %f' % (val_loss[-1]) )
        print(time.time()-start_time)
        print('*'*40)
        if len(val_loss)>10 and val_loss[-1]>val_loss[-2] and val_loss[-2]>val_loss[-3]:
            break

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqlen', default=32, type=int, help='sequence length')
    parser.add_argument('--bsz', default=32, type=int, help='batch size')
    parser.add_argument('--dropout', default=0, type=int, help='dropout rate')
    parser.add_argument('--nhidd', default=100, type=int, help='number of hidden nodes per layer')
    parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam', 'RMSprop', 'Adagrad'], help='optimizer')
    parser.add_argument('--lr', default=0.01, type=float, help='set learning rate')
    parser.add_argument('--maxiter', default=10, type=int, help='set max iteration number')
    parser.add_argument('--inp', default='../data/input.txt', type=str, help='path to data file')
    parser.add_argument('--outp', default='./', type=str, help='save model to')
    args = parser.parse_args()
    
    bat = args.bsz
    batch_size = args.seqlen
    dropout = args.dropout
    nhid = args.nhidd
    optim_type = args.optim
    lr = args.lr
    maxiter = args.maxiter
    inp = args.inp
    outp = args.outp
    
    if torch.cuda.is_available() == False:
        print('Cuda not found!')
    else:
        with open(inp,'r') as f:
            data = f.read()

        a = encoder(data)

        n_tr = int(0.8*len(data))
        tr_data, val_data = data[:n_tr], data[n_tr:]
        tr_data_, val_data_ = batchify(tr_data), batchify(val_data)

        model = rnn2(n_inp=a.leng, rnn_type='LSTM', bsz=bat, dropout=dropout).cuda()
        print(model)
        criterion = nn.CrossEntropyLoss()    
        optimizer = getattr(optim, optim_type)(model.parameters(), lr=lr)
        train_loss = []
        val_loss = []

        train(maxiter, False)
        plot_loss()

        torch.save(model.state_dict(), outp+'model_a.pth')

