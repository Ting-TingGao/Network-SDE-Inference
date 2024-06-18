import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus, Sigmoid, Softmax
from torch.autograd import Variable, grad

class SDIdifftype(MessagePassing):
    def __init__(self, model, n_f, msg_dim, ndim, delt_t,hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        super(SDIdifftype, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc_cohesion = Seq(
            Lin(1,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
            #Softplus()
        )
        for layer in self.msg_fnc_cohesion:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)
           
        self.msg_fnc_align = Seq(
            Lin(1,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
            #Softplus()
        )
        for layer in self.msg_fnc_align:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)
                
        self.msg_fnc_repulsion = Seq(
            Lin(1,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1),
            Sigmoid()
        )
        for layer in self.msg_fnc_repulsion:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)
                
        self.node_fnc_strength_x = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        for layer in self.node_fnc_strength_x:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)
        
        self.node_fnc_strength_y = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        for layer in self.node_fnc_strength_y:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std = 1e-1)

        self.stochastic_x = Seq(
            Lin(ndim*2,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_x:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
                
        self.stochastic_y = Seq(
            Lin(ndim*2,hidden),
            ReLU(),
            Lin(hidden,1),
            Softplus()
        )
        for layer in self.stochastic_y:
            if isinstance(layer,nn.Linear):
                param_shape = layer.weight.shape
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)

        
        self.stochastic_z = Seq(
            Lin(ndim*2,hidden),
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
        tmp = torch.cat([x_i, x_j], dim=1)
        if self.ndim == 1:
            xij = x_j[:,0]-x_i[:,0]
            Rij = xij.reshape(-1,1)
            vxij = x_j[:,1]-x_i[:,1]
            vij = vxij.reshape(-1,1)
            rij = torch.sqrt(xij**2)
        if self.ndim == 2:
            xij = x_j[:,0]-x_i[:,0]
            yij = x_j[:,1]-x_i[:,1]
            vxij = x_j[:,2]-x_i[:,2]
            vyij = x_j[:,3]-x_i[:,3]
            Rij = torch.cat([xij.reshape(-1,1), yij.reshape(-1,1)], dim=1)
            vij = torch.cat([vxij.reshape(-1,1), vyij.reshape(-1,1)], dim=1)
            rij = torch.sqrt(xij**2+yij**2)
            zero = torch.zeros_like(rij)
            one = torch.ones_like(rij)
            vision = torch.where(rij>0.1,zero,one)
            #print(vision)
        if self.ndim == 3:
            xij = x_j[:,0]-x_i[:,0]
            yij = x_j[:,1]-x_i[:,1]
            zij = x_j[:,2]-x_i[:,2]
            vxij = x_j[:,3]-x_i[:,3]
            vyij = x_j[:,4]-x_i[:,4]
            vzij = x_j[:,5]-x_i[:,5]
            Rij = torch.cat([xij.reshape(-1,1), yij.reshape(-1,1),zij.reshape(-1,1)], dim=1)
            vij = torch.cat([vxij.reshape(-1,1), vyij.reshape(-1,1),vzij.reshape(-1,1)], dim=1)
            rij = torch.sqrt(xij**2+yij**2+zij**2)

        Message = self.msg_fnc_cohesion(rij.reshape(-1,1))*Rij*vision.reshape(-1,1)+self.msg_fnc_align(rij.reshape(-1,1))*vij*vision.reshape(-1,1)#+self.msg_fnc_repulsion(rij.reshape(-1,1))*Rij
        #print(torch.sum(self.msg_fnc_cohesion(rij)))
        return Message
    

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            vxi = x[:,1]
            vi = vxi.reshape(-1,1)
            Vi = torch.sqrt(vxi**2)
            F = self.node_fnc_strength(Vi.reshape(-1,1))
            dvdt = F*vi+aggr_out
            v_update = x[:,self.ndim:]+dvdt*self.delt_t
            v_mean = x[:,self.ndim:]+dvdt*self.delt_t
            v_var_x = self.stochastic_x(x)
            x_update = x[:,:self.ndim]+x[:,self.ndim:]*self.delt_t+v_update*self.delt_t
            return torch.distributions.Normal(v_mean, v_var_x),x_update,v_update,dvdt
        elif self.ndim==2:
            vxi = x[:,2]
            vyi = x[:,3]
            vi = torch.cat([vxi.reshape(-1,1),vyi.reshape(-1,1)],dim=1)
            Vi = torch.sqrt(vxi**2+vyi**2)
            Vx = torch.cat([Vi.reshape(-1,1),vxi.reshape(-1,1)],dim=1)
            Vy = torch.cat([Vi.reshape(-1,1),vyi.reshape(-1,1)],dim=1)
            Fx = self.node_fnc_strength_x(Vx)
            Fy = self.node_fnc_strength_y(Vy)
            F = torch.cat((Fx,Fy),dim=1)
            #dvdt = F*vi+aggr_out
            dvdt = F+aggr_out
            v_update = x[:,self.ndim:]+dvdt*self.delt_t
            v_mean = x[:,self.ndim:]+dvdt*self.delt_t
            v_var_x = self.stochastic_x(x)
            v_var_y = self.stochastic_y(x)
            x_update = x[:,:self.ndim]+x[:,self.ndim:]*self.delt_t+v_update*self.delt_t
            return torch.distributions.Normal(v_mean[:,0].reshape(-1,1), v_var_x),torch.distributions.Normal(v_mean[:,1].reshape(-1,1), v_var_y),x_update,v_update,dvdt
        elif self.ndim==3:
            vxi = x[:,3]
            vyi = x[:,4]
            vzi = x[:,5]
            vi = torch.cat([vxi.reshape(-1,1),vyi.reshape(-1,1),vzi.reshape(-1,1)],dim=1)
            Vi = vxi**2+vyi**2+vzi**2
            F = self.node_fnc_strength(Vi.reshape(-1,1))
            dvdt = F*vi+aggr_out
            v_update = x[:,self.ndim:]+dvdt*self.delt_t
            v_mean = x[:,self.ndim:]+dvdt*self.delt_t
            v_var_x = self.stochastic_x(x)
            v_var_y = self.stochastic_y(x)
            v_var_z = self.stochastic_z(x)
            x_update = x[:,:self.ndim]+x[:,self.ndim:]*self.delt_t+v_update*self.delt_t
            return torch.distributions.Normal(v_mean[:,0].reshape(-1,1), v_var_x),torch.distributions.Normal(v_mean[:,1].reshape(-1,1), v_var_y),torch.distributions.Normal(v_mean[:,2].reshape(-1,1), v_var_z),x_update,v_update,dvdt



class SDI_Difftype(SDIdifftype):
     def __init__(
 		self, model, n_f, msg_dim, ndim, delt_t,
 		edge_index, aggr='add', hidden=50, nt=1):
            super(SDI_Difftype, self).__init__(model, n_f, msg_dim, ndim, delt_t, hidden=hidden, aggr=aggr)
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
                out_dist_x,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
                neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,1].reshape(-1,1))
                x_loss = torch.sum((g.y[:,0] - xUpdate)**2)
                v_loss = torch.sum((g.y[:,1] - vUpdate)**2)
                a_loss = torch.sum((g.y[:,2] - a_est)**2)
                dis_loss = torch.sum(neg_log_likelihood_x)
                #print(x_loss,v_loss,a_loss,dis_loss)
                return dis_loss*1e-2+a_loss*1e4#+x_loss*1e6+v_loss*1e6
            if self.ndim==2:
                out_dist_x,out_dist_y,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
                neg_log_likelihood_x = -out_dist_x.log_prob(g.y[:,2].reshape(-1,1))
                neg_log_likelihood_y = -out_dist_y.log_prob(g.y[:,3].reshape(-1,1))
                x_loss = torch.sum((g.y[:,:2] - xUpdate)**2)
                v_loss = torch.sum((g.y[:,2:4] - vUpdate)**2)
                a1_loss = torch.sum((g.y[:,4] - a_est[:,0])**2)
                a2_loss = torch.sum((g.y[:,5] - a_est[:,1])**2)
                dis_loss = torch.sum(neg_log_likelihood_x)+torch.sum(neg_log_likelihood_y)
                #print(g.y[:,6:])
                #print(a_est)
                #print(x_loss,v_loss,a1_loss,a2_loss,dis_loss)
                return dis_loss+a1_loss*3e3+a2_loss*1e3+x_loss*1e6+v_loss*1e6
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
                return dis_loss+a_loss*1e6+x_loss*1e6+v_loss*1e6
                #return dis_loss
     def sample_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                out_dist_x,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
                vxUpdate_sample = out_dist_x.sample()
                vUpdate_sample = vxUpdate_sample.reshape(-1,1)
                xUpdate_sample = g.x[:,0].reshape(-1,1)+g.x[:,1].reshape(-1,1)*delt_t+vUpdate_sample*delt_t
                return xUpdate_sample,vUpdate_sample
            if self.ndim == 2:
                out_dist_x,out_dist_y,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
                vxUpdate_sample = out_dist_x.sample()
                vyUpdate_sample = out_dist_y.sample()
                vUpdate_sample = torch.cat((vxUpdate_sample.reshape(-1,1),vyUpdate_sample.reshape(-1,1)),dim=1)
                xUpdate_sample = g.x[:,0:2].reshape(-1,2)+g.x[:,2:4].reshape(-1,2)*delt_t+vUpdate_sample*delt_t
                return xUpdate_sample,vUpdate_sample
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
                out_dist_x,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
                return xUpdate,vUpdate
            if self.ndim == 2:
                out_dist_x,out_dist_y,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
                return xUpdate,vUpdate
            if self.ndim == 3:
                out_dist_x,out_dist_y,out_dist_z,xUpdate,vUpdate,a_est = self.SDI_weighted(g)
                return xUpdate,vUpdate