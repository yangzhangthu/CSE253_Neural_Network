from mnist import MNIST
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import argparse

def data_load(str='./data/'):
    mndata = MNIST(str)
    images, labels = mndata.load_training()
    images_ts, labels_ts = mndata.load_testing()
    return images, labels, images_ts, labels_ts

def one_hot(i):
    leng = len(i)
    label = np.zeros((leng,10))
    for idx,j in enumerate(i):
        label[idx,j] = 1
    return label

def data_preproc(images, labels, images_ts, label_ts):
    data_tr = np.hstack((np.array(images[0:50000])*1.0/127.5-1,np.ones((50000,1))))
    labels_tr = one_hot(labels[0:50000])
    data_ho = np.hstack((np.array(images[50000:])*1.0/127.5-1,np.ones((10000,1))))
    labels_ho = labels[50000:]
    data_ts = np.hstack((np.array(images_ts)*1.0/127.5-1,np.ones((10000,1))))
    return data_tr, labels_tr, data_ho, labels_ho, data_ts, labels_ts

def plot_(nn1,lr):
    plt.figure()
    plt.plot(nn1.accu_[0])
    plt.plot(nn1.accu_[1])
    plt.plot(nn1.accu_[2])
    plt.grid()
    plt.title('Accuracy vs epoch with lr = '+str(lr))
    plt.legend(['train','hold-out','test'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    plt.figure()
    plt.plot(nn1.loss_[0])
    plt.plot(nn1.loss_[1])
    plt.plot(nn1.loss_[2])
    plt.grid()
    plt.title('Loss vs epoch with lr = '+str(lr))
    plt.legend(['train','hold-out','test'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

class two_nn(object):
    def __init__(self, n_hidden, batch_size, is_shuffle=False, act='lr',is_norm=True,solver='sgd'):
        self.alpha = 0.9
        self.solver = solver
        self.is_shuffle = is_shuffle
        self.actvt_fun = act
        self.is_norm = is_norm
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        if not self.is_norm:
            np.random.seed(253)
            self.wo = (np.random.rand(n_hidden+1, 10)*1-0.5)*10
            np.random.seed(253)
            self.wh = (np.random.rand(data_tr.shape[1], n_hidden)*1-0.5)*10
            np.random.seed(253)
            self.wh2 = (np.random.rand(data_tr.shape[1], data_tr.shape[1]-1)*1-0.5)*10
        else:
            np.random.seed(253)
            self.wo = (np.random.rand(n_hidden+1, 10)*1-0.5)*12.0/np.sqrt(n_hidden+1)
            np.random.seed(253)
            self.wh = (np.random.rand(data_tr.shape[1], n_hidden)*1-0.5)*12.0/np.sqrt(data_tr.shape[1])
            np.random.seed(253)
            self.wh2 = (np.random.rand(data_tr.shape[1], data_tr.shape[1]-1)*1-0.5)*12.0/np.sqrt(data_tr.shape[1])
        self.dwo = np.zeros((n_hidden+1, 10))
        self.dwh = np.zeros((data_tr.shape[1], n_hidden))
        self.dwh2 = np.zeros((data_tr.shape[1], data_tr.shape[1]-1))
        self.cur = 0
        self.epoch = 0
        self.eta = 0.01
        self.accu_ = [[],[],[]]
        self.loss_ = [[],[],[]]
    
    def fit(self, data, label, eta=0.0001,thrld=1000):
        self.data = copy.deepcopy(data)
        self.label = copy.deepcopy(label)
        epoch = 0
        self.eta = eta
        flag = 0
        ho_err = 10
        while(True):
            batch_, label_ = self.get_batch()
            y,z,x = self.forward(batch_)
            dwo,dwh,dhw2 = self.back(y, label_,z,batch_,x)
            if self.epoch > epoch:
                epoch += 1
                self.shuffle()
                l_tr,l_ho,l_ts,accu_tr,accu_ho,accu_ts = self.eval_()
                self.accu_[0].append(accu_tr)
                self.accu_[1].append(accu_ho)
                self.accu_[2].append(accu_ts)
                self.loss_[0].append(l_tr)
                self.loss_[1].append(l_ho)
                self.loss_[2].append(l_ts)
                if l_ho > ho_err:
                    flag+=1
                else:
                    flag = 0
                ho_err = l_ho
                print('Epoch'+str(epoch)+': training accuracy is '+str(accu_tr)+
                      ', loss is '+str(l_tr) + '  validation accuracy is '+str(accu_ho)+
                      ', loss is '+str(l_ho))
                if flag == 3 or epoch>thrld:
                    break
        return 0
    
    def forward(self, batch):
        c = batch.dot(self.wh2)
        x = self.act_fun(c)
        
        b = x.dot(self.wh)
        z = self.act_fun(b)
        
        a = z.dot(self.wo)
        y = self.softmax(a)
        return y,z,x
    
    def act_fun(self,b):
        a = 2*1.0/3
        if self.actvt_fun == 'sigmoid':
            temp = 1.7159*np.tanh(a*b)
        elif self.actvt_fun == 'lr':
            temp = 1.0/(1.0+np.exp(-1*b))
        elif self.actvt_fun == 'relu':
            temp = np.maximum(np.zeros(b.shape),b)
        m = temp.shape[0]
        return np.hstack((temp,np.ones((m,1))))
    
    def softmax(self, a):
        a_ = a-np.outer(np.amax(a,axis=1),np.ones((1,a.shape[1])))
        temp = np.exp(a_)
        temp2 = np.sum(temp,axis=1)
        temp3 = np.outer(temp2,np.ones((1,temp.shape[1])))
        return temp*1.0/temp3
    
    def get_batch(self):
        cur_ = self.cur
        next_ = min(self.cur + self.batch_size, self.data.shape[0])
        batch = self.data[cur_:next_, :]
        label = self.label[cur_:next_, :]
        if next_ == self.data.shape[0]:
            next_ = 0
            self.epoch += 1
        self.cur = next_
        return batch,label
    
    def back(self,y,t,z,batch,x):
        delta = t-y
        dwo = z.T.dot(delta)
        tmp = z[:,0:self.n_hidden]
        if self.actvt_fun == 'lr':
            act_prime = (tmp*(1-tmp))
        elif self.actvt_fun == 'sigmoid':
            act_prime = (1-(tmp*1.0/1.7159)**2)*1.7159*2*1.0/3
        elif self.actvt_fun == 'relu':
            act_prime = (tmp>0)
        delta_ = delta.dot(self.wo[0:self.n_hidden,:].T)*act_prime
        dwh = x.T.dot(delta_)
        
        tmp_ = x[:,0:data_tr.shape[1]-1]
        if self.actvt_fun == 'lr':
            act_prime_ = (tmp_*(1-tmp_))
        elif self.actvt_fun == 'sigmoid':
            act_prime_ = (1-(tmp_*1.0/1.7159)**2)*1.7159*2*1.0/3
        elif self.actvt_fun == 'relu':
            act_prime_ = (tmp_>0)
        delta__ = delta_.dot(self.wh[0:data_tr.shape[1]-1,:].T)*act_prime_
        dwh2 = batch.T.dot(delta__)
        
        if self.solver == 'momentum':
            self.dwo = self.eta*dwo + self.alpha*self.dwo
            self.dwh = self.eta*dwh + self.alpha*self.dwh
            self.dwh2 = self.eta*dwh2 + self.alpha*self.dwh2
            self.wo += self.dwo
            self.wh += self.dwh
            self.wh2 += self.dwh2
        elif self.solver == 'nesterov':
            tmp_dwo = copy.deepcopy(self.dwo)
            tmp_dwh = copy.deepcopy(self.dwh)
            tmp_dwh2 = copy.deepcopy(self.dwh2)
            self.dwo = self.eta*dwo + self.alpha*self.dwo
            self.dwh = self.eta*dwh + self.alpha*self.dwh
            self.dwh2 = self.eta*dwh2 + self.alpha*self.dwh2
            self.wo += (1+self.alpha)*self.dwo - self.alpha*tmp_dwo
            self.wh += (1+self.alpha)*self.dwh - self.alpha*tmp_dwh
            self.wh2 += (1+self.alpha)*self.dwh2 - self.alpha*tmp_dwh2
        elif self.solver == 'sgd':
            self.wo += self.eta*dwo
            self.wh += self.eta*dwh
            self.wh2 += self.eta*dwh2
        return dwo,dwh,dwh2
    
    def eval_(self):
        y_tr,z_tr,x_tr = self.forward(data_tr)
        y_ho,z_ho,x_ho = self.forward(data_ho)
        y_ts,z_ts,x_ts = self.forward(data_ts)
        
        l_tr = self.loss(y_tr, labels[0:50000])
        l_ho = self.loss(y_ho, labels[50000:])
        l_ts = self.loss(y_ts, labels_ts)

        accu_tr = self.accu(y_tr, labels[0:50000])
        accu_ho = self.accu(y_ho, labels[50000:])
        accu_ts = self.accu(y_ts, labels_ts)

        return l_tr,l_ho,l_ts,accu_tr,accu_ho,accu_ts
    
    def loss(self,y,t):
        loss_ = 0
        for idx,j in enumerate(t):
            loss_ -= np.log(y[idx,j])
        loss_ = loss_*1.0/len(t)
        return loss_
    
    def accu(self, y, t):
        y_ = np.argmax(y,axis=1)
        accu_ = 1 - np.count_nonzero(y_-np.array(t))*1.0/len(t)
        return accu_
    
    def test(self, mode='hidden',idx=(0,0), epsilon = 1e-2):
        
        batch_, label_ = self.get_batch()
        y,z = self.forward(batch_)
        dwo,dwh = self.back(y, label_,z,batch_)
        
        if mode=='hidden':
            self.wh[idx[0],idx[1]] += epsilon
            y,z = self.forward(batch_)
            E1 = self.loss(y,np.argmax(label_,axis=1))
            self.wh[idx[0],idx[1]] -= 2.0*epsilon
            y2,z2 = self.forward(batch_)
            E2 = self.loss(y2,np.argmax(label_,axis=1))
            
            slope = (E1-E2)*1.0/2/epsilon
            gd = -1*dwh[idx[0],idx[1]]/self.batch_size

        elif mode == 'output':

            self.wo[idx[0],idx[1]] += epsilon
            y,z = self.forward(batch_)
            E1 = self.loss(y,np.argmax(label_,axis=1))
            self.wo[idx[0],idx[1]] -= 2.0*epsilon
            y2,z2 = self.forward(batch_)
            E2 = self.loss(y2,np.argmax(label_,axis=1))
            
            slope = (E1-E2)*1.0/2/epsilon
            gd = -1*dwo[idx[0],idx[1]]/self.batch_size

        print(slope,gd)
        return 0
        
    def shuffle(self):
        if self.is_shuffle == True:
            np.random.seed(253)
            np.random.shuffle(self.data)
            np.random.seed(253)
            np.random.shuffle(self.label)
        return 0

class nn(object):
    def __init__(self, n_hidden, batch_size, is_shuffle=False, act='lr',is_norm=True,momentum=False):
        self.alpha = 0.9
        self.momentum = momentum
        self.is_shuffle = is_shuffle
        self.actvt_fun = act
        self.is_norm = is_norm
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        if not self.is_norm:
            np.random.seed(253)
            self.wo = np.random.rand(n_hidden+1, 10)*1-0.5
            np.random.seed(253)
            self.wh = np.random.rand(data_tr.shape[1], n_hidden)*1-0.5
        else:
            np.random.seed(253)
            self.wo = (np.random.rand(n_hidden+1, 10)*1-0.5)*12.0/np.sqrt(n_hidden+1)
            np.random.seed(253)
            self.wh = (np.random.rand(data_tr.shape[1], n_hidden)*1-0.5)*12.0/np.sqrt(data_tr.shape[1])
        self.dwo = np.zeros((n_hidden+1, 10))
        self.dwh = np.zeros((data_tr.shape[1], n_hidden))
        self.cur = 0
        self.epoch = 0
        self.eta = 0.01
        self.accu_ = [[],[],[]]
        self.loss_ = [[],[],[]]
    
    def fit(self, data, label, eta=0.0001,thrld=1000):
        self.data = copy.deepcopy(data)
        self.label = copy.deepcopy(label)
        epoch = 0
        self.eta = eta
        flag = 0
        ho_err = 10
        while(True):
            batch_, label_ = self.get_batch()
            y,z = self.forward(batch_)
            dwo,dwh = self.back(y, label_,z,batch_)
            if self.epoch > epoch:
                epoch += 1
                self.shuffle()
                l_tr,l_ho,l_ts,accu_tr,accu_ho,accu_ts = self.eval_()
                self.accu_[0].append(accu_tr)
                self.accu_[1].append(accu_ho)
                self.accu_[2].append(accu_ts)
                self.loss_[0].append(l_tr)
                self.loss_[1].append(l_ho)
                self.loss_[2].append(l_ts)
                if l_ho > ho_err:
                    flag+=1
                else:
                    flag = 0
                ho_err = l_ho
                print('Epoch'+str(epoch)+': training accuracy is '+str(accu_tr)+
                      ', loss is '+str(l_tr) + '  validation accuracy is '+str(accu_ho)+
                      ', loss is '+str(l_ho))
                if flag == 3 or epoch>thrld:
                    break
        return 0
    
    def forward(self, batch):
        b = batch.dot(self.wh)
        z = self.act_fun(b)
        a = z.dot(self.wo)
        y = self.softmax(a)
        return y,z
    
    def act_fun(self,b):
        a = 2*1.0/3
        if self.actvt_fun == 'sigmoid':
            temp = 1.7159*np.tanh(a*b)
        elif self.actvt_fun == 'lr':
            temp = 1.0/(1.0+np.exp(-1*b))
        m = temp.shape[0]
        return np.hstack((temp,np.ones((m,1))))
    
    def softmax(self, a):
        temp = np.exp(a)
        temp2 = np.sum(temp,axis=1)
        temp3 = np.outer(temp2,np.ones((1,temp.shape[1])))
        return temp*1.0/temp3
    
    def get_batch(self):
        cur_ = self.cur
        next_ = min(self.cur + self.batch_size, self.data.shape[0])
        batch = self.data[cur_:next_, :]
        label = self.label[cur_:next_, :]
        if next_ == self.data.shape[0]:
            next_ = 0
            self.epoch += 1
        self.cur = next_
        return batch,label
    
    def back(self,y,t,z,x):
        delta = t-y
        dwo = z.T.dot(delta)
        tmp = z[:,0:self.n_hidden]
        if self.actvt_fun == 'lr':
            act_prime = (tmp*(1-tmp))
        elif self.actvt_fun == 'sigmoid':
            act_prime = (1-(tmp*1.0/1.7159)**2)*1.7159*2*1.0/3
        delta_ = delta.dot(self.wo[0:self.n_hidden,:].T)*act_prime
        dwh = x.T.dot(delta_)
        
        if self.momentum:
            self.dwo = self.eta*dwo + self.alpha*self.dwo
            self.dwh = self.eta*dwh + self.alpha*self.dwh
            self.wo += self.dwo
            self.wh += self.dwh
        else:
            self.wo += self.eta*dwo
            self.wh += self.eta*dwh
        return dwo,dwh
    
    def eval_(self):
        y_tr,z_tr = self.forward(data_tr)
        y_ho,z_ho = self.forward(data_ho)
        y_ts,z_ts = self.forward(data_ts)
        
        l_tr = self.loss(y_tr, labels[0:50000])
        l_ho = self.loss(y_ho, labels[50000:])
        l_ts = self.loss(y_ts, labels_ts)

        accu_tr = self.accu(y_tr, labels[0:50000])
        accu_ho = self.accu(y_ho, labels[50000:])
        accu_ts = self.accu(y_ts, labels_ts)

        return l_tr,l_ho,l_ts,accu_tr,accu_ho,accu_ts
    
    def loss(self,y,t):
        loss_ = 0
        for idx,j in enumerate(t):
            loss_ -= np.log(y[idx,j])
        loss_ = loss_*1.0/len(t)
        return loss_
    
    def accu(self, y, t):
        y_ = np.argmax(y,axis=1)
        accu_ = 1 - np.count_nonzero(y_-np.array(t))*1.0/len(t)
        return accu_
    
    def test(self, mode='hidden',idx=(0,0), epsilon = 1e-2):
        
        batch_, label_ = self.get_batch()
        y,z = self.forward(batch_)
        dwo,dwh = self.back(y, label_,z,batch_)
        
        if mode=='hidden':
            self.wh[idx[0],idx[1]] += epsilon
            y,z = self.forward(batch_)
            E1 = self.loss(y,np.argmax(label_,axis=1))
            self.wh[idx[0],idx[1]] -= 2.0*epsilon
            y2,z2 = self.forward(batch_)
            E2 = self.loss(y2,np.argmax(label_,axis=1))
            
            slope = (E1-E2)*1.0/2/epsilon
            gd = -1*dwh[idx[0],idx[1]]/self.batch_size

        elif mode == 'output':

            self.wo[idx[0],idx[1]] += epsilon
            y,z = self.forward(batch_)
            E1 = self.loss(y,np.argmax(label_,axis=1))
            self.wo[idx[0],idx[1]] -= 2.0*epsilon
            y2,z2 = self.forward(batch_)
            E2 = self.loss(y2,np.argmax(label_,axis=1))
            
            slope = (E1-E2)*1.0/2/epsilon
            gd = -1*dwo[idx[0],idx[1]]/self.batch_size

        print(slope,gd)
        return 0
        
    def shuffle(self):
        if self.is_shuffle == True:
            np.random.seed(253)
            np.random.shuffle(self.data)
            np.random.seed(253)
            np.random.shuffle(self.label)
        return 0


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--layer', default=1, type=int, choices=[1, 2], help='only support 1 or 2 layer(s)')
	parser.add_argument('--nnode', default=64, type=int, help='number of hidden nodes')
	parser.add_argument('--shuffle', default=False, type=bool, choices=[True, False], help='shuffle data after each epoch')
	parser.add_argument('--activation', default='lr', type=str, choices=['lr', 'sigmoid', 'relu'], help='choose activation function from lr, sigmoid, relu')
	parser.add_argument('--wnorm', default=False, type=bool, choices=[True, False], help='use weight normalization')
	parser.add_argument('--optim', default='nesterov', type=str, choices=['nesterov', 'momentum', 'sgd'], help='choose optimizer from sgd, momentum, nesterov')
	parser.add_argument('--vanilla', default=True, type=bool, choices=[True, False], help='use momentum or vanilla')
	parser.add_argument('--lr', default=0.0001, type=float, help='set learning rate')
	parser.add_argument('--maxiter', default=5, type=int, help='set max iteration number')
	args = parser.parse_args()
	
	layer = args.layer
	nnode = args.nnode
	is_shuffle = args.shuffle
	act_fun = args.activation
	is_weight_norm = args.wnorm
	optimizer = args.optim
	is_momentum = 1-args.vanilla
	lr = args.lr
	max_iter = args.maxiter-1
	
	if layer==1 and act_fun=='relu':
		print('relu is only available in two layers currently.')
		act_fun = 'lr'
	
	print('Network is {} layer(s), {} hidden nodes, the activation function is {}'.format(layer, nnode, act_fun))
	if layer==2:
		print('Optimizer is {}' .format(optimizer))
	elif layer==1:
		print('use momentum is {}' .format(is_momentum))
	print('Shuffle data is {}, weight normalization is {},' .format(is_shuffle, is_weight_norm))
	print('learning rate is {}, max iteration number is {}'.format(lr, max_iter))
	print('-'*50)
	print('loading data')
	images, labels, images_ts, labels_ts = data_load(str='./data/')
	data_tr, labels_tr, data_ho, labels_ho, data_ts, labels_ts = data_preproc(images, labels, images_ts, labels_ts)
	print('data is loaded')
	print('-'*50)
	if layer == 1:
		nn = nn(nnode,128,is_shuffle,act_fun,is_weight_norm,is_momentum)
		nn.fit(data=data_tr, label=labels_tr, eta=lr, thrld=max_iter)
	elif layer == 2:
		nn = two_nn(nnode,128,is_shuffle,act_fun,is_weight_norm,optimizer)
		nn.fit(data=data_tr, label=labels_tr, eta=lr, thrld=max_iter)
	plot_(nn,lr)
		
