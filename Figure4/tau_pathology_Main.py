
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

"""Import model"""
import sys
sys.path.append("/home/ubuntu/GTT/StochasticDynamics")
import GNNTaupath
from GNNTaupath import *

"""Switch GPU on"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#USE_CUDA = False

"""Import data and prior information, pre-process data"""
Timeseries = pd.read_csv('/home/ubuntu/GTT/StochasticDynamics/initial_injection_dataset.csv',encoding='utf-8',header=None)
E = pd.read_csv('/home/ubuntu/GTT/StochasticDynamics/Elist_re_an_eu.csv',encoding='utf-8',header=None)

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

import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing

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
ogn = ogn.cuda()

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
        
        tmp_r = torch.cat([Tmp[:,1].reshape(-1,1),w[:,0].reshape(-1,1)],dim=1)
        m12_r = ogn.msg_fnc_ret(tmp_r)
        tmp_a = torch.cat([Tmp[:,1].reshape(-1,1),w[:,1].reshape(-1,1)],dim=1)
        m12_a = ogn.msg_fnc_ant(tmp_a)
        tmp_e = torch.cat([Tmp[:,1].reshape(-1,1),w[:,2].reshape(-1,1)],dim=1)
        m12_e = ogn.msg_fnc_euc(tmp_e)
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
    ogn.cuda()
    total_loss = 0.0
    i = 0
    num_items = 0
    while i < batch_per_epoch:
        for ginput in trainloader:
            if i >= batch_per_epoch:
                break
            opt.zero_grad()
            ginput.x = ginput.x.cuda()
            ginput.y = ginput.y.cuda()
            ginput.edge_index = ginput.edge_index.cuda()
            ginput.batch = ginput.batch.cuda()
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
ogn.cuda()
ogn.load_state_dict(recorded_models[-1])
X = torch.as_tensor(np.array(mapping_data).astype('float'))
y = torch.as_tensor(np.array(goal_data).astype('float'))
_q = Data(
    x=X[0].float().cuda(),
    edge_index=edge_index.cuda(),
    y=y[0].float().cuda())

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

true_timeseries = pd.read_csv('/home/ubuntu/GTT/StochasticDynamics/aveData.csv',encoding='utf-8',header=None)
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
plt.savefig('/home/ubuntu/GTT/StochasticDynamics/taupath_inject.pdf')
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

plt.savefig('/home/ubuntu/GTT/StochasticDynamics/taupath_accuracy_injec.pdf')
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