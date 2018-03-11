from mnist import MNIST
mndata = MNIST('./python-mnist/data/')
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
%matplotlib inline

images, labels = mndata.load_training()
images_ts, labels_ts = mndata.load_testing()

class LR(object):
    def __init__(self, train_img, train_lbl, test_img, test_lbl, pos, neg):
        tr_data = []
        tr_lbl = []
        ts_data = []
        ts_lbl = []
        for i,l in enumerate(train_lbl):
            if l==pos or l==neg:
                tr_data.append(train_img[i])
                if l==pos:
                    tr_lbl.append(1)
                else:
                    tr_lbl.append(0)
        for i,l in enumerate(test_lbl):
            if l==pos or l==neg:
                ts_data.append(test_img[i])
                if l==pos:
                    ts_lbl.append(1)
                else:
                    ts_lbl.append(0)
        hold_out = int(len(tr_data)*0.9)
        
        self.tr_data = np.array(tr_data[0:hold_out])*1.0/255
        self.tr_lbl = tr_lbl[0:hold_out]
        self.val_data = np.array(tr_data[hold_out:])*1.0/255
        self.val_lbl = tr_lbl[hold_out:]
        self.ts_data = np.array(ts_data)*1.0/255
        self.ts_lbl = ts_lbl
        
        leng = len(tr_data[0])
#         lst = np.array([random.random() for i in xrange(leng+1)])
#         lst = (lst-np.mean(lst))/np.std(lst)
        lst = np.array([0.0 for i in xrange(leng+1)])
        self.w = lst[1:]
        self.w0 = lst[0]
        
        self.cur = 0
        self.batch_size = 0
        self.batch = 0
        self.t = 0
        self.epoch = [[0],[0]]
        self.accu = [[],[],[]]
        self.loss = [[],[],[]]
    
    def fetch(self):
        if self.cur == self.tr_data.shape[0]:
            self.cur = 0
        next_ = min(self.cur+self.batch_size, self.tr_data.shape[0])
        data = self.tr_data[self.cur:next_,:]
        label = self.tr_lbl[self.cur:next_]
        self.cur = next_
        return data, label
    
    def fit(self, lr0, batch, mode, lbd, T=10000):
        flg=1
        self.batch = batch
        self.t = int(1/batch)
        self.batch_size = int(self.tr_data.shape[0]*self.batch)
        count = 1
        is_overfit = 0
        if mode == '':
            lbd = 0
        while(flg):
            
            data_,label_ = self.fetch()
            size_tr = data_.shape[0]
            
            lr = lr0*1.0/(1+count/T/self.t)
            
            b,p = self.predict(data_)
            c = np.array(label_) - b
            d = np.outer(c,np.ones((1,784)))*data_
            dw0 = sum(c)/c.shape[0]
            dw = sum(d,1)/c.shape[0]
            if mode == 'L1':
                dw0 = dw0 - lbd*float((self.w0>=0)*2-1)
                dw = dw - lbd*((self.w>=0)*2-1)
            elif mode == 'L2':
                dw0 = dw0 - 2*lbd*self.w0
                dw = dw - 2*lbd*self.w
            self.w0 = self.w0 + lr*dw0
            self.w = self.w + lr*dw
            
            self.evalu(count%self.t, mode, lbd)

            if len(self.loss[1]) < 3:
                count += 1
                continue;
            if self.loss[1][-1]>self.loss[1][-2]*0.9999:
                is_overfit += 1
            else:
                is_overfit = 0
            if is_overfit == 3:
                flg = 0
            count += 1
#             if count%100 == 0:
#                 print count
        self.epoch = [self.epoch[0][1:],self.epoch[1][1:]]
        print 'train accuracy: '+str(self.accu[0][-1])
        print 'test accuracy: '+str(self.accu[2][-1])

    def predict(self,data):
        a = data.dot(self.w)+self.w0
        b = 1/(1+np.exp(-a))
        pred = []
        for i in b:
            if i >0.5:
                pred.append(1)
            else:
                pred.append(0)
        return b,pred
    
    def evalu(self, flg, mode, lbd):
        tr_s,tr_pred = self.predict(self.tr_data)
        self.accu[0].append(self.accuracy(tr_pred,self.tr_lbl))
        self.loss[0].append(self.compute_E(tr_s,self.tr_lbl, mode, lbd))
        self.epoch[0].append(self.epoch[0][-1]+self.batch)
        
        if flg == 0:
            val_s,val_pred = self.predict(self.val_data)
            ts_s,ts_pred = self.predict(self.ts_data)
            self.accu[1].append(self.accuracy(val_pred,self.val_lbl))
            self.loss[1].append(self.compute_E(val_s,self.val_lbl, mode, lbd))
            self.accu[2].append(self.accuracy(ts_pred,self.ts_lbl))
            self.loss[2].append(self.compute_E(ts_s,self.ts_lbl, mode, lbd))
            self.epoch[1].append(self.epoch[1][-1]+1)
    
    def accuracy(self,pred,label):
        err = 0
        for i,p in enumerate(pred):
            if p-label[i]==0:
                err+=1
        return err*1.0/len(pred)
    
    def compute_E(self, scores, target, mode, lbd):
        E = 0
        for i,s in enumerate(scores):
            t = target[i]*1.0
            E -= (t*np.log(s)+(1-t)*np.log(1-s))
        E = E/len(target)
        if mode == 'L2':
            E += lbd*(self.w.dot(self.w)+self.w0**2)
        elif mode == 'L1':
            E += lbd*(sum(abs(self.w))+abs(self.w0))
        return E
    
    def plot_weights(self, t):
        plt.figure()
        w_ = self.w
        w_ = w_.reshape(28,28)
        plt.imshow(w_, cmap='binary')
        plt.title(t)
    
    def plot_accu(self, tx):
        plt.figure()
        plt.plot(self.epoch[0], self.accu[0])
        plt.plot(self.epoch[1], self.accu[1])
        plt.plot(self.epoch[1], self.accu[2])
        plt.legend(['train','vali','test'], loc=4)
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs Epoch'+tx)
    
    def plot_loss(self, tx):
        plt.figure()
        plt.plot(self.epoch[0], self.loss[0])
        plt.plot(self.epoch[1], self.loss[1])
        plt.plot(self.epoch[1], self.loss[2])
        plt.legend(['train','vali','test'],loc=1)
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss vs Epoch'+tx)

lr0 = LR(train_img=images[0:20000], train_lbl=labels[0:20000], 
         test_img=images_ts[-2000:], test_lbl=labels_ts[-2000:], pos=2, neg=3)
lr0.fit(lr0=0.1,batch=1,mode='',lbd=0.1,T=10000)
lr0.plot_loss(tx=' for 2v3')
lr0.plot_accu(tx=' for 2v3')
lr0.plot_weights(t='weights for 2v3')

lr1 = LR(train_img=images[0:20000], train_lbl=labels[0:20000], 
         test_img=images_ts[-2000:], test_lbl=labels_ts[-2000:], pos=2, neg=8)
lr1.fit(lr0=0.1,batch=1,mode='',lbd=0.1,T=10000)
lr1.plot_loss(tx=' for 2v8')
lr1.plot_accu(tx=' for 2v8')
lr1.plot_weights(t='weights for 2v8')

diff = (lr0.w-lr1.w).reshape(28,28)
plt.figure()
plt.imshow(diff, cmap='binary')
plt.title('difference between two weights')

plt.figure()
for k in xrange(1,6):
    a = 10**-k

    lr2 = LR(train_img=images[0:20000], train_lbl=labels[0:20000], 
         test_img=images_ts[-2000:], test_lbl=labels_ts[-2000:], pos=2, neg=3)
    lr2.fit(lr0=0.1,batch=1,mode='L1',lbd=a,T=10000)
    print 'lambda='+str(a)+' L1 fitted!'
    plt.plot(lr2.epoch[0], lr2.accu[0])
    
plt.legend(['l='+str(0.1),'l='+str(0.01),'l='+str(0.001),'l='+str(0.0001),'l='+str(0.00001)],loc=4)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs Epoch for L1')

plt.figure()
for k in xrange(1,6):
    a = 10**-k

    lr2 = LR(train_img=images[0:20000], train_lbl=labels[0:20000], 
         test_img=images_ts[-2000:], test_lbl=labels_ts[-2000:], pos=2, neg=3)
    lr2.fit(lr0=0.1,batch=1,mode='L2',lbd=a,T=10000)
    print 'lambda='+str(a)+' L2 fitted!'
    plt.plot(lr2.epoch[0], lr2.accu[0])
    
plt.legend(['l='+str(0.1),'l='+str(0.01),'l='+str(0.001),'l='+str(0.0001),'l='+str(0.00001)],loc=4)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs Epoch for L2')

lambda_ = []
accu_1 = []
accu_ts_1 = []
len_1 = []
accu_2 = []
accu_ts_2 = []
len_2 = []
for k in xrange(1,6):
    a = 10**-k

    lr2 = LR(train_img=images[0:20000], train_lbl=labels[0:20000], 
         test_img=images_ts[-2000:], test_lbl=labels_ts[-2000:], pos=2, neg=3)
    lr3 = LR(train_img=images[0:20000], train_lbl=labels[0:20000], 
         test_img=images_ts[-2000:], test_lbl=labels_ts[-2000:], pos=2, neg=3)
    lr2.fit(lr0=0.1,batch=1,mode='L1',lbd=a,T=10000)
    print 'lambda='+str(a)+' L1 fitted!'
    lr3.fit(lr0=0.1,batch=1,mode='L2',lbd=a,T=10000)
    print 'lambda='+str(a)+' L2 fitted!'
    
    lambda_.append(a)
    accu_1.append(lr2.accu[0][-1])
    accu_2.append(lr3.accu[0][-1])
    accu_ts_1.append(lr2.accu[2][-1])
    accu_ts_2.append(lr3.accu[2][-1])
    l1 = np.sqrt(lr2.w.dot(lr2.w)+lr2.w0**2)
    l2 = np.sqrt(lr3.w.dot(lr3.w)+lr3.w0**2)
    len_1.append(l1)
    len_2.append(l2)
    print 'iter'+str(k)+' done!'
    lr2.plot_weights(t='weights for L1 with lambda='+str(a))
    lr3.plot_weights(t='weights for L2 with lambda='+str(a))
	
plt.figure()
plt.semilogx(lambda_, accu_1)
plt.semilogx(lambda_, accu_2)
plt.legend(['L1','L2'],loc=3)
plt.title('Regurization: Lambda vs Accuracy')
plt.xlabel('lambda')
plt.ylabel('accuracy')
plt.grid()

plt.figure()
plt.semilogx(lambda_, len_1)
plt.semilogx(lambda_, len_2)
plt.legend(['L1','L2'],loc=3)
plt.title('Regurization: Lambda vs Length of Weights')
plt.xlabel('lambda')
plt.ylabel('length')
plt.grid()

plt.figure()
plt.semilogx(lambda_, accu_ts_1)
plt.semilogx(lambda_, accu_ts_2)
plt.legend(['L1','L2'],loc=3)
plt.title('Regurization: Lambda vs Test Accuracy')
plt.xlabel('lambda')
plt.ylabel('accuracy')
plt.grid()