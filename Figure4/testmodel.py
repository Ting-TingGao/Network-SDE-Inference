import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch_geometric.data import Data, DataLoader
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')



from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus
from torch.autograd import Variable, grad


"""Import model"""
class InwNGN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, weights, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        super(InwNGN, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc_ret = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )
        
        self.msg_fnc_ant = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )

        self.msg_fnc_euc = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )
        

        self.node_fnc_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )

        self.time_scale = Seq(
            Lin(1,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        Len = int(tmp.shape[0])/int(self.weights.shape[0])
        w = self.weights.repeat(int(Len),1)
        w = w.clone().detach()
        w = w.to(torch.float32)
        w = w.to(device)

        #Tmp = tmp[:,1]-tmp[:,0]

        tmp1 = torch.cat([tmp[:,1].reshape(-1,1),w[:,0].reshape(-1,1)],dim=1)
        tmp2 = torch.cat([tmp[:,1].reshape(-1,1),w[:,1].reshape(-1,1)],dim=1)
        tmp3 = torch.cat([tmp[:,1].reshape(-1,1),w[:,2].reshape(-1,1)],dim=1)
        w1 = torch.zeros_like(w)
        w1[w != 0] = 1
        w1 = w1.to(torch.float32)
        w1 = w1.to(device)
        return self.msg_fnc_ret(tmp1)*w1[:,0].reshape(-1,1)+ self.msg_fnc_ant(tmp2)*w1[:,1].reshape(-1,1) + self.msg_fnc_euc(tmp3)*w1[:,2].reshape(-1,1)


    def update(self, aggr_out, x=None):
        if self.ndim==1:
            t1 = np.full(int(x.shape[0]),1, dtype=int)
            t2 = np.full(int(x.shape[0]),3, dtype=int)
            t3 = np.full(int(x.shape[0]),6, dtype=int)
            t4 = np.full(int(x.shape[0]),9, dtype=int)
            # #t1 = t1.clone().detach()
            t1 = torch.from_numpy(t1).float()
            t1 = t1.to(device)
            # #t2 = t2.clone().detach()
            t2 = torch.from_numpy(t2).float()
            t2 = t2.to(device)
            # #t3 = t3.clone().detach()
            t3 = torch.from_numpy(t3).float()
            t3 = t3.to(device)
            # #t4 = t4.clone().detach()
            t4 = torch.from_numpy(t4).float()
            t4 = t4.to(device)

            # all1 = torch.cat([x,aggr_out,t1.reshape(-1,1)],dim=1)
            # all2 = torch.cat([x,aggr_out,t2.reshape(-1,1)],dim=1)
            # all3 = torch.cat([x,aggr_out,t3.reshape(-1,1)],dim=1)
            # all4 = torch.cat([x,aggr_out,t4.reshape(-1,1)],dim=1)
            T1 = self.time_scale(t1.reshape(-1,1))
            T2 = self.time_scale(t2.reshape(-1,1))
            T3 = self.time_scale(t3.reshape(-1,1))
            T4 = self.time_scale(t4.reshape(-1,1))



            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out


            # tmpx = torch.cat([x,aggr_out],dim=1)
            # dxdt = self.node_fnc_x(tmpx)


            y1 = dxdt*T1
            y2 = dxdt*T2
            y3 = dxdt*T3
            y4 = dxdt*T4
            return torch.cat([y1.reshape(-1,160),y2.reshape(-1,160),y3.reshape(-1,160),y4.reshape(-1,160)], dim=1)
            #dxdt = fx+aggr_out
            #return torch.cat([x+dxdt*self.delt_t,dxdt], dim=1)

            
        # only consider 1-dimensional situation

        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            return torch.cat([x_update,y_update,dxdt,dydt], dim=1)
        elif self.ndim==3:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            fz = self.node_fnc_z(x)
            dxdt = fx+aggr_out
            dydt = fy
            dzdt = fz
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            z_update = x[:,2].reshape(-1,1)+dzdt*self.delt_t
            return torch.cat([x_update,y_update,z_update,dxdt,dydt,dzdt], dim=1)


class InwNeuG(InwNGN):
     def __init__(
 		self, n_f, msg_dim, ndim, weights,
 		edge_index, aggr='add', hidden=50, nt=1):
            super(InwNeuG, self).__init__(n_f, msg_dim, ndim, weights, hidden=hidden, aggr=aggr)
            self.nt = nt
            self.edge_index = edge_index
            self.ndim = ndim
            self.weights = weights
    
     def prediction(self, g, augment=False, augmentation=3):
            #x is [n, n_f]f
            x = g.x
            ndim = self.ndim
            if augment:
                augmentation = torch.randn(1, ndim)*augmentation
                augmentation = augmentation.repeat(len(x), 1).to(x.device)
                x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
            edge_index = g.edge_index
            return self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)

    
     def loss(self, g,square=False, **kwargs):
            if square:
                return torch.sum((g.y - self.prediction(g))**2)
            else:
                return torch.sum(torch.abs(g.y.reshape(-1,640) - self.prediction(g)))
            
     def average_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                xUpdate = self.prediction(g)
                return xUpdate
# import sys
# sys.path.append("/home/ubuntu/GTT/StochasticDynamics")
# import GNNTaupath
# from GNNTaupath import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


"""Import data and prior information, pre-process data"""
Timeseries = pd.read_csv('initial_injection_dataset.csv',encoding='utf-8',header=None)
E = pd.read_csv('Elist_re_an_eu.csv',encoding='utf-8',header=None)
edge_index = E.iloc[:,0:2]-1
edge_index = torch.tensor(edge_index.values.reshape(2,-1))

W = torch.from_numpy(E.iloc[:,2:5].values.reshape(-1,3))

ts = Timeseries.values


timeseries = ts.copy()

Dimension = 1
Nodes = 160
import numpy as np
goal_data = timeseries[:,Nodes:].reshape(-1,Nodes*4,1)
mapping_data = timeseries[:,:Nodes].reshape(-1,Nodes,1)

X = torch.as_tensor(np.array(mapping_data).astype('float'))
y = torch.as_tensor(np.array(goal_data).astype('float'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
X_train = X_train.float()
y_train = y_train.float()
X_test = X_test.float()
y_test = y_test.float()

aggr = 'add'
hidden = 200
model = 'Rossler' 


msg_dim = 1
n_f = 1
dim = Dimension*1


from torch_geometric.data import Data, DataLoader

batch = 32
trainloader = DataLoader(
    [Data(
        Variable(X_train[i]),
        edge_index=edge_index,
        y=Variable(y_train[i])) for i in range(len(y_train))],
    batch_size=batch,
    shuffle=True
)

testloader = DataLoader(
    [Data(
        X_test[i],
        edge_index=edge_index,
        y=y_test[i]) for i in range(len(y_test))],
    batch_size=64,
    shuffle=True
)

ogn = InwNeuG(n_f, msg_dim, dim, W, hidden=hidden, edge_index=edge_index , aggr=aggr)#SDIweighted(model,n_f, msg_dim, dim, delt_t, W, hidden=hidden, edge_index=edge_index , aggr=aggr)

messages_over_time = []
selfDyn_over_time = []
timescale_over_time = []
ogn = ogn.to(device)

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
init_lr = 1e-5
opt = torch.optim.Adam(ogn.parameters(), lr=init_lr, weight_decay=1e-8)
total_epochs = 50 #10
batch_per_epoch = 2000#int(1000*10 / (batch/32.0))
sched = OneCycleLR(opt, max_lr=init_lr,
                   steps_per_epoch=batch_per_epoch,#len(trainloader),
                   epochs=total_epochs, final_div_factor=1e5)

"""Record self- and interaction dynamics values"""
from tqdm import tqdm
import numpy as onp
onp.random.seed(0)
test_idxes = onp.random.randint(0, len(X_test), 100)

newtestloader = DataLoader(
     [Data(
         X_test[i],
         edge_index=edge_index,
         y=y_test[i]) for i in test_idxes],
     batch_size=128,
     shuffle=False
 )


import numpy as onp
import pandas as pd

def get_messages(ogn):

    def get_message_info(tmp):
        ogn.cpu()

        s1 = tmp.x[tmp.edge_index[0]] #source
        s1 = s1[:,0]
        #print(s1)
        s2 = tmp.x[tmp.edge_index[1]] #target
        s2 = s2[:,0]
        #print(s2)
        Tmp = torch.cat([s2, s1]) # tmp --> xi,xj
        Tmp = Tmp.reshape(2,-1)
        Tmp = Tmp.t()# tmp has shape [E, 2 * in_channels]
        Tmp_new = Tmp[:,1]-Tmp[:,0]
        Len = int(s1.shape[0])/int(W.shape[0])
        w = W.repeat(int(Len),1)
        w = w.to(torch.float32)
        tmpW = torch.cat([Tmp,w],dim=1)
        tmpW = tmpW.to(torch.float32)
#         tmpW = torch.cat([s1,w[:,0]])
#         tmpW = tmpW.reshape(2,-1)
#         tmpW = tmpW.t()
#         tmpW = tmpW.to(torch.float32)
        #source_x = tmp.x[tmp.edge_index[1]]
        #tmpW = torch.cat([Tmp,source_x[:,Dimension].reshape(source_x.shape[0],1)],dim=1)
        #print(tmp)

        w1 = torch.zeros_like(w)
        w1[w != 0] = 1
        w1 = w1.to(torch.float32)
        tmp_r = torch.cat([Tmp[:,1].reshape(-1,1),w[:,0].reshape(-1,1)],dim=1)
        m12_r = ogn.msg_fnc_ret(tmp_r)#*w1[:,0].reshape(-1,1)
        tmp_a = torch.cat([Tmp[:,1].reshape(-1,1),w[:,1].reshape(-1,1)],dim=1)
        m12_a = ogn.msg_fnc_ant(tmp_a)#*w1[:,1].reshape(-1,1)
        tmp_e = torch.cat([Tmp[:,1].reshape(-1,1),w[:,2].reshape(-1,1)],dim=1)
        m12_e = ogn.msg_fnc_euc(tmp_e)#*w1[:,2].reshape(-1,1)
        #/source_x[:,Dimension].reshape(source_x.shape[0],1)

        all_messages = torch.cat((
            tmpW,
             m12_r,m12_a,m12_e), dim=1)
        if dim == 1:
            columns = [elem%(k) for k in range(1,3) for elem in 'x%d'.split(' ')]
            columns += ['wr','wa','we']
            columns += ['er','ea','ee']
        if dim == 2:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        elif dim == 3:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['e%d'%(k,) for k in range(msg_dim)]

        return pd.DataFrame(
              data=all_messages.cpu().detach().numpy(),
             columns=columns)


    msg_info = []
    for i, g in enumerate(newtestloader):
        msg_info.append(get_message_info(g))

    msg_info = pd.concat(msg_info)
    return msg_info


def get_selfDynamics(ogn):
    def get_selfDynamics_info(tmp):
        ogn.cpu()
        
        tmp = tmp.x[tmp.edge_index[1]]
        tmp = tmp[:,0:Dimension].reshape(-1,Dimension)
        if dim==1:
            self_dyn_x = ogn.node_fnc_x(tmp)
            self_dyn_all = torch.cat((tmp,self_dyn_x), dim=1)
            columns = ['x','s1']
            
        if dim==2:
            self_dyn_x = ogn.node_fnc_x(tmp)
            self_dyn_y = ogn.node_fnc_y(tmp)
            self_dyn_all = torch.cat((tmp,self_dyn_x,self_dyn_y), dim=1)
            columns = ['x','y','s1','s2']
        if dim==3:
            self_dyn_x = ogn.node_fnc_x(tmp)
            self_dyn_y = ogn.node_fnc_y(tmp)
            self_dyn_z = ogn.node_fnc_z(tmp)
            self_dyn_all = torch.cat((tmp,self_dyn_x,self_dyn_y,self_dyn_z), dim=1)
            columns = ['x','y','z','s1','s2','s3']
            
        return pd.DataFrame(
              data=self_dyn_all.cpu().detach().numpy(),
             columns=columns
        )

    selfDyn_info = []
    for i, g in enumerate(newtestloader):
        selfDyn_info.append(get_selfDynamics_info(g))

    selfDyn_info = pd.concat(selfDyn_info)
    return selfDyn_info

def get_timescale(ogn):
    def get_timescale_info(tmp):
        ogn.cpu()
        
        xtmp = tmp.x[tmp.edge_index[1]]
        t1 = np.full(int(xtmp.shape[0]),1, dtype=int)
        t2 = np.full(int(xtmp.shape[0]),3, dtype=int)
        t3 = np.full(int(xtmp.shape[0]),6, dtype=int)
        t4 = np.full(int(xtmp.shape[0]),9, dtype=int)
            # #t1 = t1.clone().detach()
        t1 = torch.from_numpy(t1).float()
        #t1 = t1.cuda()
            # #t2 = t2.clone().detach()
        t2 = torch.from_numpy(t2).float()
        #t2 = t2.cuda()
            # #t3 = t3.clone().detach()
        t3 = torch.from_numpy(t3).float()
        #t3 = t3.cuda()
            # #t4 = t4.clone().detach()
        t4 = torch.from_numpy(t4).float()
        #t4 = t4.cuda()
        if dim==1:
            T1 = ogn.time_scale(t1.reshape(-1,1))
            T2 = ogn.time_scale(t2.reshape(-1,1))
            T3 = ogn.time_scale(t3.reshape(-1,1))
            T4 = ogn.time_scale(t4.reshape(-1,1))
            self_diff_all = torch.cat((T1,T2,T3,T4), dim=1)
            columns = ['t1','t3','t6','t9']
            
            
        return pd.DataFrame(
              data=self_diff_all.cpu().detach().numpy(),
             columns=columns
        )

    selfTimescale_info = []
    for i, g in enumerate(newtestloader):
        selfTimescale_info.append(get_timescale_info(g))

    selfTimescale_info = pd.concat(selfTimescale_info)
    return selfTimescale_info


"""Start training"""
epoch = 0
recorded_models = []

for epoch in tqdm(range(epoch, total_epochs)):
    ogn.to(device)
    total_loss = 0.0
    i = 0
    num_items = 0
    while i < batch_per_epoch:
        for ginput in trainloader:
            if i >= batch_per_epoch:
                break
            opt.zero_grad()
            ginput.x = ginput.x.to(device)
            ginput.y = ginput.y.to(device)
            ginput.edge_index = ginput.edge_index.to(device)
            ginput.batch = ginput.batch.to(device)
            loss = ogn.loss(ginput)
            (loss/int(ginput.batch[-1]+1)).backward()
            opt.step()
            sched.step()

            total_loss += loss.item()
            i += 1
            num_items += int(ginput.batch[-1]+1)


    cur_loss = total_loss/num_items
    print(cur_loss)
    cur_msgs = get_messages(ogn)
    cur_selfdyn = get_selfDynamics(ogn)
    cur_diff = get_timescale(ogn)
    cur_msgs['epoch'] = epoch
    cur_msgs['loss'] = cur_loss
    messages_over_time.append(cur_msgs)
    selfDyn_over_time.append(cur_selfdyn)
    timescale_over_time.append(cur_diff)
    
    ogn.cpu()
    from copy import deepcopy as copy
    recorded_models.append(ogn.state_dict())
"""Reproduce the trajectories"""
ogn.to(device)
ogn.load_state_dict(recorded_models[-1])
X = torch.as_tensor(np.array(mapping_data).astype('float'))
y = torch.as_tensor(np.array(goal_data).astype('float'))
_q = Data(
    x=X[0].float().to(device),
    edge_index=edge_index.to(device),
    y=y[0].float().to(device))

x_i = ogn.average_trajectories(_q)


import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt


# fig = plt.figure(figsize=(5,8))
# ax1 = fig.add_subplot(2,1,1)
# ax1.plot(x_tra)
# plt.ylabel('inferred trajectory')
# ax2 = fig.add_subplot(2,1,2)
# t = np.arange(0,x_tra.shape[0])
# ax2.plot(t,x_tra, t, x_real[1:x_tra.shape[0]+1,])
# plt.ylabel('real vs inferred trajectory')
# plt.savefig('/home/ubuntu/GTT/StochasticDynamics/delta_trajectories.png')
# plt.close()


#time1 = y[0,:,:].reshape(1,-1)

true_timeseries = pd.read_csv('aveData.csv',encoding='utf-8',header=None)
time1 = true_timeseries.values.reshape(-1,4,160)


plt.rcParams.update({'font.size': 12}) 
plt.rcParams.update({'font.style': 'normal'})
# plt.rcParams.update({'font.family': 'Arial'})
# mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300
fig = plt.figure(figsize=(16,5))
x_i = x_i.cpu()
x_itmp = x_i.detach().numpy()
x_tra = [x_itmp[0,0],x_itmp[0,160],x_itmp[0,160*2],x_itmp[0,160*3]]
#x_real = [time1[0,0],time1[0,160],time1[0,160*2],time1[0,160*3]]
x_real = time1[0,0:,0]*1
print(x_tra,x_real)

ax1 = fig.add_subplot(1,3,1)
t = np.arange(0,4)
ax1.plot(t,x_tra,c='k',label="inferred")
ax1.plot(t, x_real,c='b',label="real")
plt.ylabel('real vs inferred trajectory of dimension x')
plt.legend()
plt.savefig('taupath_inject.pdf')
plt.close()

sx = x_i.detach().numpy()[0,0:160]
sx_true = time1[0,0,:]
sy = x_i.detach().numpy()[0,160:160*2]
sy_true = time1[0,1,:]
sz = x_i.detach().numpy()[0,160*2:160*3]
sz_true = time1[0,2,:]
sn = x_i.detach().numpy()[0,160*3:160*4]
sn_true = time1[0,3,:]


fig = plt.figure(figsize=(20, 4))
ax1 = fig.add_subplot(1,4,1)
plt.title("1MPI",fontsize=10)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
ax1.scatter(np.log(sx_true),np.log(sx),s=20,c ='steelblue', alpha=0.2)
ax1.plot((0, 1), (0, 1), transform=ax1.transAxes, ls='--',c='k', label="1:1 line")
# parameterx = np.polyfit(sx_true,sx,1)
# fx = np.poly1d(parameterx)
#ax1.plot(sx_true,fx(sx_true),c='palevioletred',lw=1.5)
corrx = np.corrcoef(sx_true,sx)[0,1]
bbox = dict(fc='1',alpha=0.5)
plt.text(0.05, 0.9, '$R=%.2f$' % (corrx), transform=ax1.transAxes, size=15, bbox=bbox)
plt.xlabel("True")
plt.ylabel("Inferred")
plt.xlim((-10, 5))
plt.ylim((-10, 5))

ax2 = fig.add_subplot(1,4,2)
plt.title("3MPI",fontsize=10)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
ax2.scatter(np.log(sy_true),np.log(sy),s=20,c ='steelblue', alpha=0.2)
ax2.plot((0, 1), (0, 1), transform=ax2.transAxes, ls='--',c='k', label="1:1 line")
# parametery = np.polyfit(sy_true,sy,1)
# fy = np.poly1d(parametery)
#ax2.plot(sy_true,fy(sy_true),c='palevioletred',lw=1.5)
corry = np.corrcoef(sy_true,sy)[0,1]
bbox = dict(fc='1',alpha=0.5)
plt.text(0.05, 0.9, '$R=%.2f$' % (corry), transform=ax2.transAxes, size=15, bbox=bbox)
plt.xlabel("True")
plt.ylabel("Inferred")
plt.xlim((-10, 5))
plt.ylim((-10, 5))

ax3 = fig.add_subplot(1,4,3)
plt.title("6MPI",fontsize=10)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
ax3.scatter(np.log(sz_true),np.log(sz),s=20,c ='steelblue', alpha=0.2)
ax3.plot((0, 1), (0, 1), transform=ax3.transAxes, ls='--',c='k', label="1:1 line")
# parameterz = np.polyfit(sz_true,sz,1)
# fz = np.poly1d(parameterz)
#ax3.plot(sz_true,fz(sz_true),c='palevioletred',lw=1.5)
corrz = np.corrcoef(sz_true,sz)[0,1]
bbox = dict(fc='1',alpha=0.5)
plt.text(0.05, 0.9, '$R=%.2f$' % (corrz), transform=ax3.transAxes, size=15, bbox=bbox)
plt.xlabel("True")
plt.ylabel("Inferred")
plt.xlim((-10, 5))
plt.ylim((-10, 5))

ax4 = fig.add_subplot(1,4,4)
plt.title("9MPI",fontsize=10)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
ax4.scatter(np.log(sn_true),np.log(sn),s=20,c ='steelblue', alpha=0.2)
ax4.plot((0, 1), (0, 1), transform=ax4.transAxes, ls='--',c='k', label="1:1 line")
# parameterz = np.polyfit(sz_true,sz,1)
# fz = np.poly1d(parameterz)
#ax3.plot(sz_true,fz(sz_true),c='palevioletred',lw=1.5)
corrn = np.corrcoef(sn_true,sn)[0,1]
bbox = dict(fc='1',alpha=0.5)
plt.text(0.05, 0.9, '$R=%.2f$' % (corrn), transform=ax4.transAxes, size=15, bbox=bbox)
plt.xlabel("True")
plt.ylabel("Inferred")
plt.xlim((-10, 5))
plt.ylim((-10, 5))

plt.savefig('taupath_accuracy_homogeneous.pdf')
plt.close()

best_message = messages_over_time[-1]
bestMe = best_message
bestMe.to_csv('tauPath_interaction.csv')



best_self = selfDyn_over_time[-1]
bestS = best_self
bestS.to_csv('tauPath_self.csv')

best_time = timescale_over_time[-1]
bestT = best_time
bestT.to_csv('tauPath_timescale.csv')

torch.save(recorded_models[-1],'tauPath_reconstruction_model.pth')

