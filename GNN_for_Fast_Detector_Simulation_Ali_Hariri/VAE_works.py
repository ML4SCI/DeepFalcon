
  
#from diff_encoder import Encoder, GNN
from torch_geometric.utils import to_dense_batch, to_dense_adj
import os.path as osp
from math import ceil
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool,dense_mincut_pool
import torchvision
import torch_geometric
import torch_geometric.nn as tnn

from torch_geometric.nn import EdgeConv, NNConv, GraphConv, DenseGCNConv
from torch_geometric.nn.pool.edge_pool import EdgePooling

from torch_geometric.nn.inits import reset
from torch_geometric.nn import TopKPooling, GCNConv,GatedGraphConv, SAGPooling
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x,a,s):
        return self.lambd(x,a,s)

class GraphAE(torch.nn.Module):
    def __init__(self,in_channels, out_channels1, out_channels2,out_channels3,out_channels4, out_channels5,out_channels6, dropout):
        super(GraphAE, self).__init__()   
        maxNodes = 1500
        GraphAE.reset_parameters(self)


        #GAE.reset_parameters(self)

        self.out_channels2=out_channels2
        
        """
        Encoding
        """
        ### Encoding
        """
        self.sage1=tnn.DenseGCNConv(in_channels,out_channels1)
        self.sage2=tnn.DenseGCNConv(out_channels1,out_channels2)
        self.sage3=tnn.DenseGCNConv(out_channels2,out_channels3)
        self.sage4=tnn.DenseGCNConv(out_channels3,out_channels4)

        self.sage5=tnn.DenseGCNConv(out_channels4,out_channels5)
        """
        #DenseSAGEConv(in_channels, out_channels, normalize=False, bias=True)
        self.sage1=tnn.DenseSAGEConv(in_channels,out_channels1)
        self.sage2=tnn.DenseSAGEConv(out_channels1,out_channels2)

        self.sage3=tnn.DenseSAGEConv(out_channels2,out_channels3)
        self.sage4=tnn.DenseSAGEConv(out_channels3,out_channels4)
        self.sage42=nn.Linear(out_channels3,out_channels4)

        self.sage5=tnn.DenseSAGEConv(out_channels4,out_channels5)
        
        self.poolit1=tnn.DenseSAGEConv(out_channels2,400)
        self.poolit2=tnn.DenseSAGEConv(out_channels4,200)
        self.poolit3=tnn.DenseSAGEConv(out_channels5,10)
        

        self.tr1=nn.Linear(out_channels5,out_channels6)
        self.tr2=nn.Linear(out_channels6,64)

        self.sage22=nn.Linear(out_channels1,out_channels2)
        self.tr2=nn.Linear(out_channels6,64)


        self.rev1=nn.Linear(out_channels6,out_channels5)
        self.rev2=nn.Linear(64,out_channels6)
        
        """
        self.revsage1=tnn.DenseGCNConv(out_channels1,in_channels)
        self.revsage2=tnn.DenseGCNConv(out_channels2,out_channels1)

        self.revsage3=tnn.DenseGCNConv(out_channels3,out_channels2)
        self.revsage4=tnn.DenseGCNConv(out_channels4,out_channels3)

        self.revsage5=tnn.DenseGCNConv(out_channels5,out_channels4)
        """
        self.revsage1=tnn.DenseSAGEConv(out_channels1,in_channels)
        self.revsage2=tnn.DenseSAGEConv(out_channels2,out_channels1)
        self.revsage22=nn.Linear(out_channels2,out_channels1)

        self.revsage3=tnn.DenseSAGEConv(out_channels3,out_channels2)
        self.revsage4=tnn.DenseSAGEConv(out_channels4,out_channels3)
        self.revsage42=nn.Linear(out_channels4,out_channels3)

        self.revsage5=tnn.DenseSAGEConv(out_channels5,out_channels4)

        self.drop5=torch.nn.Dropout(p=0.5)
        self.drop4=torch.nn.Dropout(p=0.4)
        self.drop3=torch.nn.Dropout(p=0.3)

        ## Batch Normalization
        self.bano1 = nn.BatchNorm1d(num_features=1000)
        self.bano2 = nn.BatchNorm1d(num_features=1000)
        self.bano3 = nn.BatchNorm1d(num_features=400)
        self.bano4 = nn.BatchNorm1d(num_features=400)
        self.bano5 = nn.BatchNorm1d(num_features=200)
        self.bano6 = nn.BatchNorm1d(num_features=10)

        #self.upsample2=LambdaLayer(self.upsample())
    

    def reset_parameters(self):
        reset(self.encode)
        reset(self.decode)

    def upsample(self,X,A,S):
      
      Xout=torch.bmm(S,X)
      Aout=torch.bmm(S,torch.bmm(S,A).permute(0,2,1))
      return Xout,Aout

    def upsample2(self,X,A,S):
      trans=torchvision.transforms.Lambda()
      
      return trans(self.upsample(X,A,S))

    def encode(self,whole,wholeAdj,lengs,refMat,maxNodes):  
        ### 1 
        hidden1=self.sage1(whole,wholeAdj)
        hidden1=F.tanh(hidden1) ## BxNxL1
        hidden1=self.bano1(hidden1)
        hidden1=self.drop5(hidden1)
      
        ### 2
        hidden2=self.sage22(hidden1)#,wholeAdj)
        hidden2=F.relu(hidden2) ## BxNxL2
        hidden2=self.bano2(hidden2)
        hidden2=self.drop4(hidden2)

        ### Pool1
        pool1=self.poolit1(hidden2,wholeAdj)
        pool1=F.relu(pool1) ## BxNxC1
 
        out1,adj1,_,_=dense_mincut_pool(hidden2,wholeAdj,pool1)
           
        ### 3
        hidden3=self.sage3(out1,adj1)
        hidden3=F.relu(hidden3)        
        hidden3=self.bano3(hidden3)
        hidden3=self.drop3(hidden3)
        
        ##hidden3=self.drop(hidden3)

        ### 4 
        hidden4=self.sage42(hidden3)#,adj1)
        
        hidden4=F.tanh(hidden4) 
        hidden4=self.bano4(hidden4)
        hidden4=self.drop3(hidden4)

        ### Pool2
        pool2=self.poolit2(hidden4,adj1)
        pool2=F.leaky_relu(pool2) ## BxN/4xC2

        out2,adj2,_,_=dense_mincut_pool(hidden4,adj1,pool2)

        out2=self.sage5(out2,adj2)
        out2=F.tanh(out2) 
        out2=self.bano5(out2)
        out2=self.drop3(out2)

        """
        ### Pool3
        pool3=self.poolit3(out2,adj2)
        pool3=F.leaky_relu(pool3) ## BxN/8xC3

        out3,adj3,_,_=dense_diff_pool(out2,adj2,pool3)
        """
        ### 5
        hidden5=self.tr1(out2)
        hidden5=F.relu(hidden5) 
        hidden5=self.drop3(hidden5)
 
        return self.tr2(hidden5),self.tr2(hidden5),adj2


    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    """
    def reparametrize(self, mu, logvar):
      if self.training:
          return mu + torch.randn_like(logvar) * torch.exp(logvar)
      else:
          return mu  
    """
    
    def decode(self,z,adj,maxNodes):

        out1=self.rev2(z)  
        out1=F.tanh(out1)   
        out1=self.drop3(out1)
        out1=self.rev1(out1)
        out1=F.tanh(out1)
        out1=self.drop3(out1) 

        """"
        s0=torch.nn.Parameter(torch.randn(out1.shape[0],200,out1.shape[1])).cuda()
        
        out1,aout0=self.upsample(out1,adj,s0)
        out1=F.tanh(out1)
        """

        out1=self.revsage5( out1,adj)
        out1=F.tanh(out1)
        #out1=self.bano5(out1)
        out1=self.drop3(out1)

        s=torch.nn.Parameter(torch.randn(out1.shape[0],400,out1.shape[1]),requires_grad=True).cuda()
        torch.nn.init.xavier_uniform(s)
        #ss=LambdaLayer(lambda s: s ** 2)

        xout2,aout2=self.upsample(out1,adj,s)
        #xout2=F.leaky_relu(xout2)

        out2=self.revsage42(xout2)#,aout2)
        out2=F.tanh(out2)
        #out2=self.bano4(out2)
        out2=self.drop3(out2)

        out2=self.revsage3(out2,aout2)
        out2=F.tanh(out2)
        #out2=self.bano4(out2)
        out2=self.drop4(out2)

        s2=torch.nn.Parameter(torch.randn(out2.shape[0],maxNodes,out2.shape[1]),requires_grad=True).cuda()
        torch.nn.init.xavier_uniform(s2)
        #ss2=LambdaLayer(lambda s: s ** 2)

        out3,aout3=self.upsample(out2,aout2,s2)
        #out3=F.leaky_relu(out3)

        out3=self.revsage22(out3)#,aout3)
        out3=F.tanh(out3)
        #out3=self.bano1(out3)
        out3=self.drop4(out3)

        out3=self.revsage1(out3,aout3)
        out3=F.tanh(out3)
        out3=self.bano1(out3)

        return out3,aout3

    def forward(self,x,adj,lengs,refMat,maxNodes):
        self.maxNodes=maxNodes
        mu,logvar,adjMat = self.encode(x,adj,lengs,refMat,maxNodes)     ## mu, log sigma 
        z = self.reparametrize(mu, logvar) ## z = mu + eps*sigma 
        z2,adj2=self.decode(z,adjMat,maxNodes)
        return z2, adj2, mu, logvar 

