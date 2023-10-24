"""Models,
   author: Ting-Ting Gao"""

import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus, Sigmoid, Softmax
from torch.autograd import Variable, grad

"""
n_f: number of features;
msg_dim: message dimensions;
ndim: dimensions of system, for example, for Hindmarsh-Rose model, ndim=3;

"""

"""NGN is the base of unweighted network dynamics (deterministic) inference GNNs"""
class NGN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i, otherwise is (i,j)'"""

        super(NGN, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
            Lin(2,hidden),
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

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        if self.ndim==1:
            tmp = torch.cat([x_i,x_j], dim=1)
        else:
            tmp = torch.cat([x_i[:,0], x_j[:,0]]) # tmp has shape [E, 2 * in_channels]
            tmp = tmp.reshape(2,-1)
            tmp = tmp.t()
        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            return x+dxdt*self.delt_t
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            return torch.cat([x_update,y_update], dim=1)
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


class NeuG(NGN):
     def __init__(
 		self, n_f, msg_dim, ndim, delt_t,
 		edge_index, aggr='add', hidden=50, nt=1):
            super(NeuG, self).__init__(n_f, msg_dim, ndim, delt_t, hidden=hidden, aggr=aggr)
            self.delt_t = delt_t
            self.nt = nt
            self.edge_index = edge_index
            self.ndim = ndim
    
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
                return torch.sum(torch.abs(g.y - self.prediction(g)))



"""wNGN is the base of weighted network dynamics inference"""
class wNGN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, weights, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        super(wNGN, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
            Lin(2,hidden),
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

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        Len = int(x_i[:,0].shape[0])/int(self.weights.shape[0])
        w = self.weights.repeat(int(Len),1)
        w = w.clone().detach()
        return self.msg_fnc(tmp)*w

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            return torch.cat([x+dxdt*self.delt_t,dxdt], dim=1)
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


class wNeuG(wNGN):
     def __init__(
 		self, n_f, msg_dim, ndim, delt_t, weights,
 		edge_index, aggr='add', hidden=50, nt=1):
            super(wNeuG, self).__init__(n_f, msg_dim, ndim, delt_t, weights, hidden=hidden, aggr=aggr)
            self.delt_t = delt_t
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
                return torch.sum(torch.abs(g.y - self.prediction(g)))



"""
************************

Structure: SDIunweighted
SDIunweighted is a network structure for identification of stochastic dynamics on unweighted network

Layers weights' std: set for different dynamics

************************
"""
class SDI(MessagePassing):
    def __init__(self, model, n_f, msg_dim, ndim, delt_t, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        super(SDI, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )
        for layer in self.msg_fnc:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)


        self.node_fnc_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        for layer in self.node_fnc_x:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
        
        self.node_fnc_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        for layer in self.node_fnc_y:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
        
        self.node_fnc_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        if model == 'HR':
            for layer in self.node_fnc_z:
                if isinstance(layer,nn.Linear):
                    param_shape = layer.weight.shape
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
        else:
            for layer in self.node_fnc_z:
                if isinstance(layer,nn.Linear):
                    param_shape = layer.weight.shape
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
        
        self.stochastic_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_x:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
        
        self.stochastic_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_y:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)

        
        self.stochastic_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_z:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):

        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        return self.msg_fnc(tmp)
    

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            x_update = x+dxdt*self.delt_t
            x_mean = x+dxdt*self.delt_t
            x_var = self.stochastic_x(x)
            return torch.distributions.Normal(x_mean, x_var),x_update
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            x_mean = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_mean = x[:,1].reshape(-1,1)+dydt*self.delt_t
            x_var = self.stochastic_x(x)
            y_var = self.stochastic_y(x)
            return torch.distributions.Normal(x_mean, x_var),torch.distributions.Normal(y_mean, y_var),x_update,y_update
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
            x_mean = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_mean = x[:,1].reshape(-1,1)+dydt*self.delt_t
            z_mean = x[:,2].reshape(-1,1)+dzdt*self.delt_t
            x_var = self.stochastic_x(x)
            y_var = self.stochastic_y(x)
            z_var = self.stochastic_z(x)
            return torch.distributions.Normal(x_mean, x_var),torch.distributions.Normal(y_mean, y_var),torch.distributions.Normal(z_mean, z_var),x_update,y_update,z_update


class SDIunweighted(SDI):
     def __init__(
 		self, model, n_f, msg_dim, ndim, delt_t, 
         edge_index, aggr='add', hidden=50, nt=1):
            super(SDIunweighted, self).__init__(model, n_f, msg_dim, ndim, delt_t, hidden=hidden, aggr=aggr)
            self.delt_t = delt_t
            self.nt = nt
            self.edge_index = edge_index
            self.ndim = ndim
    
     def SDI_unweighted(self, g, augment=False, augmentation=3):
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

     def loss(self, g, **kwargs):
            if self.ndim==1:
                out_dist,xUpdate = self.SDI_unweighted(g)
                neg_log_likelihood = -out_dist.log_prob(g.y)
                return torch.sum(neg_log_likelihood)
            if self.ndim==2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_unweighted(g)
                neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,0].reshape(-1,1))
                neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,1].reshape(-1,1))
                return torch.sum(neg_log_likelihood_x)+torch.sum(neg_log_likelihood_y)
            if self.ndim==3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,yUpdate,zUpdate = self.SDI_unweighted(g)
                neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,0].reshape(-1,1))
                neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,1].reshape(-1,1))
                neg_log_likelihood_z = -out_dist_z.log_prob(g.y[:,2].reshape(-1,1))
                return torch.mean(neg_log_likelihood_x)+torch.mean(neg_log_likelihood_y)+torch.mean(neg_log_likelihood_z)

     def squareloss(self, g,square=False, **kwargs):
            out_dist,xUpdate = self.SDI_unweighted(g)
            neg_log_likelihood = -out_dist.log_prob(g.y[:,self.ndim:])
            if square:
                #print(torch.sum(torch.abs(g.y[:,0:self.ndim])))
                #print(torch.sum(neg_log_likelihood))
                return torch.sum((g.y[:,0:self.ndim] - xUpdate)**2)+torch.sum(neg_log_likelihood)
            else:
                #print(torch.sum(torch.abs(g.y[:,0:self.ndim])))
                #print(torch.sum(neg_log_likelihood))
                return torch.sum(torch.abs(g.y[:,0:self.ndim] - xUpdate))+torch.sum(neg_log_likelihood)

     def sample_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                out_dist,xUpdate = self.SDI_unweighted(g)
                xUpdate_sample = out_dist.sample()
                return xUpdate_sample
            if self.ndim == 2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_unweighted(g)
                xUpdate_sample = out_dist_x.sample()
                yUpdate_sample = out_dist_y.sample()
                return xUpdate_sample, yUpdate_sample
            if self.ndim == 3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,yUpdate,zUpdate = self.SDI_unweighted(g)
                xUpdate_sample = out_dist_x.sample()
                yUpdate_sample = out_dist_y.sample()
                zUpdate_sample = out_dist_z.sample()
                return xUpdate_sample, yUpdate_sample, zUpdate_sample

     def average_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                out_dist,xUpdate = self.SDI_unweighted(g)
                return xUpdate
            if self.ndim == 2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_unweighted(g)
                return xUpdate, yUpdate
            if self.ndim == 3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,yUpdate,zUpdate = self.SDI_unweighted(g)
                return xUpdate, yUpdate, zUpdate


"""
Structure: SDIweighted
SDIweighted is a network structure for identification of stochastic dynamics on weighted network

Layers weights' std: set for different dynamics

"""
class SDIw(MessagePassing):
    def __init__(self, model, n_f, msg_dim, ndim, delt_t, weights, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        super(SDIw, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )
        for layer in self.msg_fnc:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)


        self.node_fnc_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        for layer in self.node_fnc_x:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
        
        self.node_fnc_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        for layer in self.node_fnc_y:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
        
        self.node_fnc_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        if model == 'HR':
            for layer in self.node_fnc_z:
                if isinstance(layer,nn.Linear):
                    param_shape = layer.weight.shape
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
        else:
            for layer in self.node_fnc_z:
                if isinstance(layer,nn.Linear):
                    param_shape = layer.weight.shape
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
        
        self.stochastic_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_x:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
        
        self.stochastic_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_y:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)

        
        self.stochastic_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_z:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):

        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        Len = int(x_j[:,0].shape[0])/int(self.weights.shape[0])
        w = self.weights.repeat(int(Len),1)
        w = w.clone().detach()
        return self.msg_fnc(tmp)*w
    

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            x_update = x+dxdt*self.delt_t
            x_mean = x+dxdt*self.delt_t
            x_var = self.stochastic_x(x)
            return torch.distributions.Normal(x_mean, x_var),x_update
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            x_mean = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_mean = x[:,1].reshape(-1,1)+dydt*self.delt_t
            x_var = self.stochastic_x(x)
            y_var = self.stochastic_y(x)
            return torch.distributions.Normal(x_mean, x_var),torch.distributions.Normal(y_mean, y_var),x_update,y_update
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
            x_mean = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_mean = x[:,1].reshape(-1,1)+dydt*self.delt_t
            z_mean = x[:,2].reshape(-1,1)+dzdt*self.delt_t
            x_var = self.stochastic_x(x)
            y_var = self.stochastic_y(x)
            z_var = self.stochastic_z(x)
            return torch.distributions.Normal(x_mean, x_var),torch.distributions.Normal(y_mean, y_var),torch.distributions.Normal(z_mean, z_var),x_update,y_update,z_update


class SDIweighted(SDIw):
     def __init__(
 		self, model, n_f, msg_dim, ndim, delt_t,weights,
 		edge_index, aggr='add', hidden=50, nt=1):
            super(SDIweighted, self).__init__(model, n_f, msg_dim, ndim, delt_t, weights, hidden=hidden, aggr=aggr)
            self.delt_t = delt_t
            self.nt = nt
            self.edge_index = edge_index
            self.ndim = ndim
            self.weights = weights
    
     def SDI_weighted(self, g, augment=False, augmentation=3):
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

     def loss(self, g, **kwargs):
            if self.ndim==1:
                out_dist,xUpdate = self.SDI_weighted(g)
                neg_log_likelihood = -out_dist.log_prob(g.y)
                return torch.sum(neg_log_likelihood)
            if self.ndim==2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_weighted(g)
                neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,0].reshape(-1,1))
                neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,1].reshape(-1,1))
                return torch.sum(neg_log_likelihood_x)+torch.sum(neg_log_likelihood_y)
            if self.ndim==3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,yUpdate,zUpdate = self.SDI_weighted(g)
                neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,0].reshape(-1,1))
                neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,1].reshape(-1,1))
                neg_log_likelihood_z = -out_dist_z.log_prob(g.y[:,2].reshape(-1,1))
                return torch.mean(neg_log_likelihood_x)+torch.mean(neg_log_likelihood_y)+torch.mean(neg_log_likelihood_z)
     def sample_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                out_dist,xUpdate = self.SDI_weighted(g)
                xUpdate_sample = out_dist.sample()
                return xUpdate_sample
            if self.ndim == 2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_weighted(g)
                xUpdate_sample = out_dist_x.sample()
                yUpdate_sample = out_dist_y.sample()
                return xUpdate_sample, yUpdate_sample
            if self.ndim == 3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,yUpdate,zUpdate = self.SDI_weighted(g)
                xUpdate_sample = out_dist_x.sample()
                yUpdate_sample = out_dist_y.sample()
                zUpdate_sample = out_dist_z.sample()
                return xUpdate_sample, yUpdate_sample, zUpdate_sample

     def average_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                out_dist,xUpdate = self.SDI_weighted(g)
                return xUpdate
            if self.ndim == 2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_weighted(g)
                return xUpdate, yUpdate
            if self.ndim == 3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,yUpdate,zUpdate = self.SDI_weighted(g)
                return xUpdate, yUpdate, zUpdate


""""""
class SDIdifftype(MessagePassing):
    def __init__(self, model, n_f, msg_dim, ndim, delt_t, Type, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        super(SDIdifftype, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc_excit = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )
        for layer in self.msg_fnc_excit:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)
        
        self.msg_fnc_inh = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )
        for layer in self.msg_fnc_inh:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)

        

        self.node_fnc_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        for layer in self.node_fnc_x:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
        
        self.node_fnc_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        for layer in self.node_fnc_y:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
        
        self.node_fnc_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        if model == 'HR':
            for layer in self.node_fnc_z:
                if isinstance(layer,nn.Linear):
                    param_shape = layer.weight.shape
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
        else:
            for layer in self.node_fnc_z:
                if isinstance(layer,nn.Linear):
                    param_shape = layer.weight.shape
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
        
        self.stochastic_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_x:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
        
        self.stochastic_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_y:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)

        
        self.stochastic_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_z:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):

        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        Len = int(x_j[:,0].shape[0])/int(self.Type.shape[0])
        T = self.Type.repeat(int(Len),1)
        T = T.clone().detach()
        Message_excit = self.msg_fnc_excit(tmp)
        Message_inh = self.msg_fnc_inh(tmp)
        T_excit = torch.where(T>0,T,0)
        T_inh = torch.where(T<0,T,0)
        Message = Message_excit*T_excit+Message_inh*T_inh
        return Message
        # for i in range(T.shape[0]):
        #     if T[i] > 0:
        #         msg_tmp = self.msg_fnc_excit(tmp[i,:])
        #     else:
        #         msg_tmp = self.msg_fnc_inh(tmp[i,:])
        #         if i == 0:
        #             Message = msg_tmp
        #         else:
        #             Message = torch.cat((Message,msg_tmp),0)
        # return Message
    

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            x_update = x+dxdt*self.delt_t
            x_mean = x+dxdt*self.delt_t
            x_var = self.stochastic_x(x)
            return torch.distributions.Normal(x_mean, x_var),x_update
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            x_mean = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_mean = x[:,1].reshape(-1,1)+dydt*self.delt_t
            x_var = self.stochastic_x(x)
            y_var = self.stochastic_y(x)
            return torch.distributions.Normal(x_mean, x_var),torch.distributions.Normal(y_mean, y_var),x_update,y_update
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
            x_mean = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_mean = x[:,1].reshape(-1,1)+dydt*self.delt_t
            z_mean = x[:,2].reshape(-1,1)+dzdt*self.delt_t
            x_var = self.stochastic_x(x)
            y_var = self.stochastic_y(x)
            z_var = self.stochastic_z(x)
            return torch.distributions.Normal(x_mean, x_var),torch.distributions.Normal(y_mean, y_var),torch.distributions.Normal(z_mean, z_var),x_update,y_update,z_update


class SDI_Difftype(SDIdifftype):
     def __init__(
 		self, model, n_f, msg_dim, ndim, delt_t,Type,
 		edge_index, aggr='add', hidden=50, nt=1):
            super(SDI_Difftype, self).__init__(model, n_f, msg_dim, ndim, delt_t, Type, hidden=hidden, aggr=aggr)
            self.delt_t = delt_t
            self.nt = nt
            self.edge_index = edge_index
            self.ndim = ndim
            self.Type = Type
    
     def SDI_weighted(self, g, augment=False, augmentation=3):
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

     def loss(self, g, **kwargs):
            if self.ndim==1:
                out_dist,xUpdate = self.SDI_weighted(g)
                neg_log_likelihood = -out_dist.log_prob(g.y)
                return torch.sum(neg_log_likelihood)
            if self.ndim==2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_weighted(g)
                neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,0].reshape(-1,1))
                neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,1].reshape(-1,1))
                return torch.sum(neg_log_likelihood_x)+torch.sum(neg_log_likelihood_y)
            if self.ndim==3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,yUpdate,zUpdate = self.SDI_weighted(g)
                neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,0].reshape(-1,1))
                neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,1].reshape(-1,1))
                neg_log_likelihood_z = -out_dist_z.log_prob(g.y[:,2].reshape(-1,1))
                return torch.sum(neg_log_likelihood_x)+torch.sum(neg_log_likelihood_y)+torch.sum(neg_log_likelihood_z)
     def sample_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                out_dist,xUpdate = self.SDI_weighted(g)
                xUpdate_sample = out_dist.sample()
                return xUpdate_sample
            if self.ndim == 2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_weighted(g)
                xUpdate_sample = out_dist_x.sample()
                yUpdate_sample = out_dist_y.sample()
                return xUpdate_sample, yUpdate_sample
            if self.ndim == 3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,yUpdate,zUpdate = self.SDI_weighted(g)
                xUpdate_sample = out_dist_x.sample()
                yUpdate_sample = out_dist_y.sample()
                zUpdate_sample = out_dist_z.sample()
                return xUpdate_sample, yUpdate_sample, zUpdate_sample

     def average_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                out_dist,xUpdate = self.SDI_weighted(g)
                return xUpdate
            if self.ndim == 2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_weighted(g)
                return xUpdate, yUpdate
            if self.ndim == 3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,yUpdate,zUpdate = self.SDI_weighted(g)
                return xUpdate, yUpdate, zUpdate
            
# underdamped Langevin equation (flocks)
class SDIunder(MessagePassing):
    def __init__(self, model, n_f, msg_dim, ndim, delt_t,hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        super(SDIunder, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc_cohesion = Seq(
            Lin(1,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)  
        )
        for layer in self.msg_fnc_cohesion:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 2e-1)
           
        self.msg_fnc_align = Seq(
            Lin(1,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        for layer in self.msg_fnc_align:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 2e-1)
                
        self.node_fnc_strength = Seq(
            Lin(1,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
            #Softmax()
        )
        
        for layer in self.node_fnc_strength:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 2e-1)

        self.stochastic_x = Seq(
            Lin(ndim*2,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_x:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
                
        self.stochastic_y = Seq(
            Lin(ndim*2,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_y:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)

        
        self.stochastic_z = Seq(
            Lin(ndim*2,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_z:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-1)
               

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)
        xij = x_j[:,0]-x_i[:,0]
        yij = x_j[:,1]-x_i[:,1]
        zij = x_j[:,2]-x_i[:,2]
        vxij = x_j[:,3]-x_i[:,3]
        vyij = x_j[:,4]-x_i[:,4]
        vzij = x_j[:,5]-x_i[:,5]
        Rij = torch.cat([xij.reshape(-1,1), yij.reshape(-1,1),zij.reshape(-1,1)], dim=1)
        vij = torch.cat([vxij.reshape(-1,1), vyij.reshape(-1,1),vzij.reshape(-1,1)], dim=1)
        rij = torch.sqrt(xij**2+yij**2+zij**2)

        Message = self.msg_fnc_cohesion(rij.reshape(-1,1))*Rij+self.msg_fnc_align(rij.reshape(-1,1))*vij
        #print(torch.sum(self.msg_fnc_cohesion(rij)))
        return Message
    

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            x_update = x+dxdt*self.delt_t
            x_mean = x+dxdt*self.delt_t
            x_var = self.stochastic_x(x)
            return torch.distributions.Normal(x_mean, x_var),x_update
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            x_mean = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_mean = x[:,1].reshape(-1,1)+dydt*self.delt_t
            x_var = self.stochastic_x(x)
            y_var = self.stochastic_y(x)
            return torch.distributions.Normal(x_mean, x_var),torch.distributions.Normal(y_mean, y_var),x_update,y_update
        elif self.ndim==3:
            vxi = x[:,3]
            vyi = x[:,4]
            vzi = x[:,5]
            vi = torch.cat([vxi.reshape(-1,1),vyi.reshape(-1,1),vzi.reshape(-1,1)],dim=1)
            Vi = torch.sqrt(vxi**2+vyi**2+vzi**2)
            F = self.node_fnc_strength(Vi.reshape(-1,1))
            dvdt = F*vi+aggr_out
            v_update = x[:,self.ndim:]+dvdt*self.delt_t
            v_mean = x[:,self.ndim:]+dvdt*self.delt_t
            v_var_x = self.stochastic_x(x)
            v_var_y = self.stochastic_y(x)
            v_var_z = self.stochastic_z(x)
            x_update = x[:,:self.ndim]+x[:,self.ndim:]*self.delt_t+v_update*self.delt_t
            return torch.distributions.Normal(v_mean[:,0].reshape(-1,1), v_var_x),torch.distributions.Normal(v_mean[:,1].reshape(-1,1), v_var_y),torch.distributions.Normal(v_mean[:,2].reshape(-1,1), v_var_z),x_update,v_update,dvdt
            #return x_update,y_update,z_update,fx


class SDI_underdamp(SDIunder):
     def __init__(
 		self, model, n_f, msg_dim, ndim, delt_t,
 		edge_index, aggr='add', hidden=50, nt=1):
            super(SDI_underdamp, self).__init__(model, n_f, msg_dim, ndim, delt_t, hidden=hidden, aggr=aggr)
            self.delt_t = delt_t
            self.nt = nt
            self.edge_index = edge_index
            self.ndim = ndim

    
     def SDI_weighted(self, g, augment=False, augmentation=3):
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

     def loss(self, g, **kwargs):
            if self.ndim==1:
                out_dist,xUpdate = self.SDI_weighted(g)
                neg_log_likelihood = -out_dist.log_prob(g.y)
                return torch.sum(neg_log_likelihood)
            if self.ndim==2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_weighted(g)
                neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,0].reshape(-1,1))
                neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,1].reshape(-1,1))
                return torch.sum(neg_log_likelihood_x)+torch.sum(neg_log_likelihood_y)
            if self.ndim==3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
                neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,3].reshape(-1,1))
                neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,4].reshape(-1,1))
                neg_log_likelihood_z = -out_dist_x.log_prob(g.y[:,5].reshape(-1,1))
                x_loss = torch.sum((g.y[:,:3] - xUpdate)**2)
                v_loss = torch.sum((g.y[:,3:6] - vUpdate)**2)
                a_loss = torch.sum((g.y[:,6:] - a_est)**2)
                dis_loss = torch.sum(neg_log_likelihood_x)+torch.sum(neg_log_likelihood_y)+torch.sum(neg_log_likelihood_z)
                #print(g.y[:,6:])
                #print(a_est)
                #print(x_loss,v_loss,a_loss,dis_loss)
                return dis_loss+x_loss*100+v_loss*100+a_loss*0.1
                #return dis_loss*10+x_loss*100+v_loss*20+a_loss*0.01
                #return dis_loss
     def sample_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                out_dist,xUpdate = self.SDI_weighted(g)
                xUpdate_sample = out_dist.sample()
                return xUpdate_sample
            if self.ndim == 2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_weighted(g)
                xUpdate_sample = out_dist_x.sample()
                yUpdate_sample = out_dist_y.sample()
                return xUpdate_sample, yUpdate_sample
            if self.ndim == 3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
                vxUpdate_sample = out_dist_x.sample()
                vyUpdate_sample = out_dist_y.sample()
                vzUpdate_sample = out_dist_z.sample()
                vUpdate_sample = torch.cat((vxUpdate_sample.reshape(-1,1),vyUpdate_sample.reshape(-1,1),vzUpdate_sample.reshape(-1,1)),dim=1)
                xUpdate_sample = g.x[:,0:3].reshape(-1,3)+g.x[:,3:6].reshape(-1,3)*delt_t+vUpdate_sample*delt_t
                return xUpdate_sample,vUpdate_sample
     def average_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                out_dist,xUpdate = self.SDI_weighted(g)
                return xUpdate
            if self.ndim == 2:
                out_dist_x,out_dist_y,xUpdate,yUpdate = self.SDI_weighted(g)
                return xUpdate, yUpdate
            if self.ndim == 3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
                return xUpdate,vUpdate
            


"""Empirical flocking"""
# class SDIdifftype(MessagePassing):
#     def __init__(self, model, n_f, msg_dim, ndim, delt_t,hidden=50, aggr='add', flow='source_to_target'):

#         """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
#         super(SDIdifftype, self).__init__(aggr=aggr, flow=flow)
#         self.msg_fnc_cohesion = Seq(
#             Lin(1,hidden),
#             ReLU(),
#             Lin(hidden,hidden),
#             ReLU(),
#             Lin(hidden,hidden),
#             ReLU(),
#             Lin(hidden,1)
#             #Softplus()
#         )
#         for layer in self.msg_fnc_cohesion:
#             if isinstance(layer,nn.Linear):
#                 param_shape = layer.weight.shape
#                 torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)
           
#         self.msg_fnc_align = Seq(
#             Lin(1,hidden),
#             ReLU(),
#             Lin(hidden,hidden),
#             ReLU(),
#             Lin(hidden,hidden),
#             ReLU(),
#             Lin(hidden,1)
#             #Softplus()
#         )
#         for layer in self.msg_fnc_align:
#             if isinstance(layer,nn.Linear):
#                 param_shape = layer.weight.shape
#                 torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)
                
#         self.msg_fnc_repulsion = Seq(
#             Lin(1,hidden),
#             ReLU(),
#             Lin(hidden,hidden),
#             ReLU(),
#             Lin(hidden,hidden),
#             ReLU(),
#             Lin(hidden,1),
#             Sigmoid()
#         )
#         for layer in self.msg_fnc_repulsion:
#             if isinstance(layer,nn.Linear):
#                 param_shape = layer.weight.shape
#                 torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)
                
#         self.node_fnc_strength_x = Seq(
#             Lin(2,hidden),
#             ReLU(),
#             Lin(hidden,hidden),
#             ReLU(),
#             Lin(hidden,hidden),
#             ReLU(),
#             Lin(hidden,1)
#         )
        
#         for layer in self.node_fnc_strength_x:
#             if isinstance(layer,nn.Linear):
#                 param_shape = layer.weight.shape
#                 torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)
        
#         self.node_fnc_strength_y = Seq(
#             Lin(2,hidden),
#             ReLU(),
#             Lin(hidden,hidden),
#             ReLU(),
#             Lin(hidden,hidden),
#             ReLU(),
#             Lin(hidden,1)
#         )
        
#         for layer in self.node_fnc_strength_y:
#             if isinstance(layer,nn.Linear):
#                 param_shape = layer.weight.shape
#                 torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)

#         self.stochastic_x = Seq(
#             Lin(ndim*2,hidden),
#             ReLU(),
#             Lin(hidden,1),
#             Softplus()
#         )
#         for layer in self.stochastic_x:
#             if isinstance(layer,nn.Linear):
#                 param_shape = layer.weight.shape
#                 torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
                
#         self.stochastic_y = Seq(
#             Lin(ndim*2,hidden),
#             ReLU(),
#             Lin(hidden,1),
#             Softplus()
#         )
#         for layer in self.stochastic_y:
#             if isinstance(layer,nn.Linear):
#                 param_shape = layer.weight.shape
#                 torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)

        
#         self.stochastic_z = Seq(
#             Lin(ndim*2,hidden),
#             ReLU(),
#             Lin(hidden,1),
#             Softplus()
#         )
#         for layer in self.stochastic_z:
#             if isinstance(layer,nn.Linear):
#                 param_shape = layer.weight.shape
#                 torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
               

#     def forward(self, x, edge_index):
#         # x has shape [N, number_of_features]
#         # edge_index has shape [2,E]
#         x = x
#         return self.propagate(edge_index, x=x)

#     def message(self, x_i, x_j):
#         tmp = torch.cat([x_i, x_j], dim=1)
#         if self.ndim == 1:
#             xij = x_j[:,0]-x_i[:,0]
#             Rij = xij.reshape(-1,1)
#             vxij = x_j[:,1]-x_i[:,1]
#             vij = vxij.reshape(-1,1)
#             rij = torch.sqrt(xij**2)
#         if self.ndim == 2:
#             xij = x_j[:,0]-x_i[:,0]
#             yij = x_j[:,1]-x_i[:,1]
#             vxij = x_j[:,2]-x_i[:,2]
#             vyij = x_j[:,3]-x_i[:,3]
#             Rij = torch.cat([xij.reshape(-1,1), yij.reshape(-1,1)], dim=1)
#             vij = torch.cat([vxij.reshape(-1,1), vyij.reshape(-1,1)], dim=1)
#             rij = torch.sqrt(xij**2+yij**2)
#             zero = torch.zeros_like(rij)
#             one = torch.ones_like(rij)
#             vision = torch.where(rij>0.1,zero,one)
#             #print(vision)
#         if self.ndim == 3:
#             xij = x_j[:,0]-x_i[:,0]
#             yij = x_j[:,1]-x_i[:,1]
#             zij = x_j[:,2]-x_i[:,2]
#             vxij = x_j[:,3]-x_i[:,3]
#             vyij = x_j[:,4]-x_i[:,4]
#             vzij = x_j[:,5]-x_i[:,5]
#             Rij = torch.cat([xij.reshape(-1,1), yij.reshape(-1,1),zij.reshape(-1,1)], dim=1)
#             vij = torch.cat([vxij.reshape(-1,1), vyij.reshape(-1,1),vzij.reshape(-1,1)], dim=1)
#             rij = torch.sqrt(xij**2+yij**2+zij**2)

#         Message = self.msg_fnc_cohesion(rij.reshape(-1,1))*Rij*vision.reshape(-1,1)+self.msg_fnc_align(rij.reshape(-1,1))*vij*vision.reshape(-1,1)#+self.msg_fnc_repulsion(rij.reshape(-1,1))*Rij
#         #print(torch.sum(self.msg_fnc_cohesion(rij)))
#         return Message
    

#     def update(self, aggr_out, x=None):
#         if self.ndim==1:
#             vxi = x[:,1]
#             vi = vxi.reshape(-1,1)
#             Vi = torch.sqrt(vxi**2)
#             F = self.node_fnc_strength(Vi.reshape(-1,1))
#             dvdt = F*vi+aggr_out
#             v_update = x[:,self.ndim:]+dvdt*self.delt_t
#             v_mean = x[:,self.ndim:]+dvdt*self.delt_t
#             v_var_x = self.stochastic_x(x)
#             x_update = x[:,:self.ndim]+x[:,self.ndim:]*self.delt_t+v_update*self.delt_t
#             return torch.distributions.Normal(v_mean, v_var_x),x_update,v_update,dvdt
#         elif self.ndim==2:
#             vxi = x[:,2]
#             vyi = x[:,3]
#             vi = torch.cat([vxi.reshape(-1,1),vyi.reshape(-1,1)],dim=1)
#             Vi = torch.sqrt(vxi**2+vyi**2)
#             Vx = torch.cat([Vi.reshape(-1,1),vxi.reshape(-1,1)],dim=1)
#             Vy = torch.cat([Vi.reshape(-1,1),vyi.reshape(-1,1)],dim=1)
#             Fx = self.node_fnc_strength_x(Vx)
#             Fy = self.node_fnc_strength_y(Vy)
#             F = torch.cat((Fx,Fy),dim=1)
#             #dvdt = F*vi+aggr_out
#             dvdt = F+aggr_out
#             v_update = x[:,self.ndim:]+dvdt*self.delt_t
#             v_mean = x[:,self.ndim:]+dvdt*self.delt_t
#             v_var_x = self.stochastic_x(x)
#             v_var_y = self.stochastic_y(x)
#             x_update = x[:,:self.ndim]+x[:,self.ndim:]*self.delt_t+v_update*self.delt_t
#             return torch.distributions.Normal(v_mean[:,0].reshape(-1,1), v_var_x),torch.distributions.Normal(v_mean[:,1].reshape(-1,1), v_var_y),x_update,v_update,dvdt
#         elif self.ndim==3:
#             vxi = x[:,3]
#             vyi = x[:,4]
#             vzi = x[:,5]
#             vi = torch.cat([vxi.reshape(-1,1),vyi.reshape(-1,1),vzi.reshape(-1,1)],dim=1)
#             Vi = vxi**2+vyi**2+vzi**2
#             F = self.node_fnc_strength(Vi.reshape(-1,1))
#             dvdt = F*vi+aggr_out
#             v_update = x[:,self.ndim:]+dvdt*self.delt_t
#             v_mean = x[:,self.ndim:]+dvdt*self.delt_t
#             v_var_x = self.stochastic_x(x)
#             v_var_y = self.stochastic_y(x)
#             v_var_z = self.stochastic_z(x)
#             x_update = x[:,:self.ndim]+x[:,self.ndim:]*self.delt_t+v_update*self.delt_t
#             return torch.distributions.Normal(v_mean[:,0].reshape(-1,1), v_var_x),torch.distributions.Normal(v_mean[:,1].reshape(-1,1), v_var_y),torch.distributions.Normal(v_mean[:,2].reshape(-1,1), v_var_z),x_update,v_update,dvdt



# class SDI_Difftype(SDIdifftype):
#      def __init__(
#  		self, model, n_f, msg_dim, ndim, delt_t,
#  		edge_index, aggr='add', hidden=50, nt=1):
#             super(SDI_Difftype, self).__init__(model, n_f, msg_dim, ndim, delt_t, hidden=hidden, aggr=aggr)
#             self.delt_t = delt_t
#             self.nt = nt
#             self.edge_index = edge_index
#             self.ndim = ndim

    
#      def SDI_weighted(self, g, augment=False, augmentation=3):
#             #x is [n, n_f]f
#             x = g.x
#             ndim = self.ndim
#             if augment:
#                 augmentation = torch.randn(1, ndim)*augmentation
#                 augmentation = augmentation.repeat(len(x), 1).to(x.device)
#                 x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
#             edge_index = g.edge_index
#             return self.propagate(
#                     edge_index, size=(x.size(0), x.size(0)),
#                     x=x)

#      def loss(self, g, **kwargs):
#             if self.ndim==1:
#                 out_dist_x,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
#                 neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,1].reshape(-1,1))
#                 x_loss = torch.sum((g.y[:,0] - xUpdate)**2)
#                 v_loss = torch.sum((g.y[:,1] - vUpdate)**2)
#                 a_loss = torch.sum((g.y[:,2] - a_est)**2)
#                 dis_loss = torch.sum(neg_log_likelihood_x)
#                 #print(x_loss,v_loss,a_loss,dis_loss)
#                 return dis_loss*1e-2+a_loss*1e4#+x_loss*1e6+v_loss*1e6
#             if self.ndim==2:
#                 out_dist_x,out_dist_y,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
#                 neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,2].reshape(-1,1))
#                 neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,3].reshape(-1,1))
#                 x_loss = torch.sum((g.y[:,:2] - xUpdate)**2)
#                 v_loss = torch.sum((g.y[:,2:4] - vUpdate)**2)
#                 a1_loss = torch.sum((g.y[:,4] - a_est[:,0])**2)
#                 a2_loss = torch.sum((g.y[:,5] - a_est[:,1])**2)
#                 dis_loss = torch.sum(neg_log_likelihood_x)+torch.sum(neg_log_likelihood_y)
#                 #print(g.y[:,6:])
#                 #print(a_est)
#                 #print(x_loss,v_loss,a1_loss,a2_loss,dis_loss)
#                 return dis_loss+a1_loss*3e3+a2_loss*1e3+x_loss*1e6+v_loss*1e6
#             if self.ndim==3:
#                 out_dist_x,out_dist_y,out_dist_z,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
#                 neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,3].reshape(-1,1))
#                 neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,4].reshape(-1,1))
#                 neg_log_likelihood_z = -out_dist_x.log_prob(g.y[:,5].reshape(-1,1))
#                 x_loss = torch.sum((g.y[:,:3] - xUpdate)**2)
#                 v_loss = torch.sum((g.y[:,3:6] - vUpdate)**2)
#                 a_loss = torch.sum((g.y[:,6:] - a_est)**2)
#                 dis_loss = torch.sum(neg_log_likelihood_x)+torch.sum(neg_log_likelihood_y)+torch.sum(neg_log_likelihood_z)
#                 #print(g.y[:,6:])
#                 #print(a_est)
#                 #print(x_loss,v_loss,a_loss,dis_loss)
#                 return dis_loss+a_loss*1e6+x_loss*1e6+v_loss*1e6
#                 #return dis_loss
#      def sample_trajectories(self, g, **kwargs):
#             if self.ndim == 1:
#                 out_dist_x,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
#                 vxUpdate_sample = out_dist_x.sample()
#                 vUpdate_sample = vxUpdate_sample.reshape(-1,1)
#                 xUpdate_sample = g.x[:,0].reshape(-1,1)+g.x[:,1].reshape(-1,1)*delt_t+vUpdate_sample*delt_t
#                 return xUpdate_sample,vUpdate_sample
#             if self.ndim == 2:
#                 out_dist_x,out_dist_y,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
#                 vxUpdate_sample = out_dist_x.sample()
#                 vyUpdate_sample = out_dist_y.sample()
#                 vUpdate_sample = torch.cat((vxUpdate_sample.reshape(-1,1),vyUpdate_sample.reshape(-1,1)),dim=1)
#                 xUpdate_sample = g.x[:,0:2].reshape(-1,2)+g.x[:,2:4].reshape(-1,2)*delt_t+vUpdate_sample*delt_t
#                 return xUpdate_sample,vUpdate_sample
#             if self.ndim == 3:
#                 out_dist_x,out_dist_y,out_dist_z,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
#                 vxUpdate_sample = out_dist_x.sample()
#                 vyUpdate_sample = out_dist_y.sample()
#                 vzUpdate_sample = out_dist_z.sample()
#                 vUpdate_sample = torch.cat((vxUpdate_sample.reshape(-1,1),vyUpdate_sample.reshape(-1,1),vzUpdate_sample.reshape(-1,1)),dim=1)
#                 xUpdate_sample = g.x[:,0:3].reshape(-1,3)+g.x[:,3:6].reshape(-1,3)*delt_t+vUpdate_sample*delt_t
#                 return xUpdate_sample,vUpdate_sample
#      def average_trajectories(self, g, **kwargs):
#             if self.ndim == 1:
#                 out_dist_x,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
#                 return xUpdate,vUpdate
#             if self.ndim == 2:
#                 out_dist_x,out_dist_y,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
#                 return xUpdate,vUpdate
#             if self.ndim == 3:
#                 out_dist_x,out_dist_y,out_dist_z,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
#                 return xUpdate,vUpdate