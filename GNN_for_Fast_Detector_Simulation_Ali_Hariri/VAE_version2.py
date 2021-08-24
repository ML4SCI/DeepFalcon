
  
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
        self.sage1=tnn.DenseSAGEConv(in_channels,out_channels1,normalize=True)
        #self.sage2=tnn.DenseSAGEConv(out_channels1,out_channels2,normalize=True)

        self.sage3=tnn.DenseSAGEConv(out_channels1,out_channels3,normalize=True)
        #self.sage4=tnn.DenseSAGEConv(out_channels3,out_channels4,normalize=True)
        ##self.sage42=nn.Linear(out_channels3,out_channels4)

        self.sage5=tnn.DenseSAGEConv(out_channels3,out_channels5,normalize=True)
        self.sage6=tnn.DenseSAGEConv(out_channels5,out_channels6,normalize=True)
        
        ##self.poolit1=tnn.DenseSAGEConv(out_channels2,400)
        self.poolit1=nn.Linear(out_channels1,250)
        self.poolit2=nn.Linear(out_channels3,50)
        #self.poolit3=nn.Linear(out_channels5,50)
        ##self.poolit2=tnn.DenseSAGEConv(out_channels4,200)
        #self.poolit3=tnn.DenseSAGEConv(out_channels5,10)
        

        #self.tr1=nn.Linear(out_channels5,out_channels6)
        #self.tr2=nn.Linear(out_channels5,64)

        self.tr2=nn.Linear(out_channels5,16)

        self.rev2=nn.Linear(16,out_channels5)
        
        """
        self.revsage1=tnn.DenseGCNConv(out_channels1,in_channels)
        self.revsage2=tnn.DenseGCNConv(out_channels2,out_channels1)

        self.revsage3=tnn.DenseGCNConv(out_channels3,out_channels2)
        self.revsage4=tnn.DenseGCNConv(out_channels4,out_channels3)

        self.revsage5=tnn.DenseGCNConv(out_channels5,out_channels4)
        """
        self.revsage1=tnn.DenseSAGEConv(out_channels1,in_channels,normalize=False)
        #self.revsage2=tnn.DenseSAGEConv(out_channels2,out_channels1,normalize=True)

        self.revsage3=tnn.DenseSAGEConv(out_channels3,out_channels1,normalize=False)

        self.revsage5=tnn.DenseSAGEConv(out_channels5,out_channels3,normalize=False)
        self.revsage6=tnn.DenseSAGEConv(out_channels6,out_channels5,normalize=False)

        self.drop5=torch.nn.Dropout(p=0.5)
        self.drop4=torch.nn.Dropout(p=0.4)
        self.drop3=torch.nn.Dropout(p=0.3)

        ## Batch Normalization
        self.bano1 = nn.BatchNorm1d(num_features=1000)
        self.bano2 = nn.BatchNorm1d(num_features=1000)
        self.bano3 = nn.BatchNorm1d(num_features=250)
        self.bano4 = nn.BatchNorm1d(num_features=50)
        self.bano5 = nn.BatchNorm1d(num_features=50)
        self.bano6 = nn.BatchNorm1d(num_features=50)
        
        #self.prelu=nn.PReLU()

    def upsample(self,X,A,S):
      Xout=torch.bmm(S,X)

      Aout=torch.bmm(S,torch.bmm(A,S.permute(0,2,1)))
      return Xout,Aout

    def encode(self,whole,adj,lengs,mask,maxNodes):  
        ### 1 
        hidden=self.sage1(whole,adj)
        hidden=F.relu(hidden)
        hidden=self.bano1(hidden)
        hidden=self.drop5(hidden)
        """
        ### 2
        hidden=self.sage2(hidden,adj)
        hidden=F.relu(hidden) 
        hidden=self.bano2(hidden)
        hidden=self.drop3(hidden)
        """

        ### Pool1
        pool1=self.poolit1(hidden)
 
        hidden,adj,mc1,o1=dense_mincut_pool(hidden,adj,pool1,mask)

           
        ### 3
        hidden=self.sage3(hidden,adj)
        hidden=F.relu(hidden)        
        hidden=self.bano3(hidden)
        hidden=self.drop4(hidden)

        ### Pool2
        pool2=self.poolit2(hidden)

        hidden,adj,mc2,o2=dense_mincut_pool(hidden,adj,pool2)

        hidden=self.sage5(hidden,adj)
        hidden=F.relu(hidden) 
        hidden=self.bano5(hidden)
        hidden=self.drop3(hidden)

        ### Pool3
        #pool3=self.poolit3(hidden)

        #hidden,adj,mc3,o3=dense_mincut_pool(hidden,adj,pool3)
        """
        hidden=self.sage6(hidden,adj)
        hidden=F.tanh(hidden) 
        hidden=self.bano6(hidden)
        hidden=self.drop3(hidden)
        """

        return self.tr2(hidden),self.tr2(hidden), adj,pool1,pool2,mc1+mc2,o1+o2


    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self,z,adj,s1,s2,maxNodes):

        out1=self.rev2(z)  
        out1=F.leaky_relu(out1)   
        out1=self.drop3(out1)
        """
        out1=self.revsage6(out1,adj)
        out1=F.leaky_relu(out1)
        out1=self.bano6(out1)
        out1=self.drop3(out1)
        """
        """
        out1,adj=self.upsample(out1,adj,s3)
        out1=F.leaky_relu(out1)
        adj=F.sigmoid(adj)
        """

        out1=self.revsage5(out1,adj)
        out1=F.relu(out1)
        out1=self.bano5(out1)
        out1=self.drop3(out1)

        out1,adj=self.upsample(out1,adj,s2)
        out1=F.leaky_relu(out1)
        adj=F.sigmoid(adj)

        out1=self.revsage3(out1,adj)
        out1=F.leaky_relu(out1)
        out1=self.bano3(out1)
        out1=self.drop4(out1)

        out1,adj=self.upsample(out1,adj,s1)
        out1=F.leaky_relu(out1)
        adj=F.sigmoid(adj)

        """
        out1=self.revsage2(out1,adj)
        out1=F.relu(out1)
        #out1=self.bano1(out1)
        out1=self.drop4(out1)
        """

        out1=self.revsage1(out1,adj)
        out1=F.relu(out1)
        #out1=self.bano1(out1)

        return out1,adj

    def forward(self,x,adj,lengs,refMat,maxNodes):
        mu,logvar,adjMat,s1,s2,l1,l2 = self.encode(x,adj,lengs,refMat,maxNodes)     ## mu, log sigma 
        z = self.reparametrize(mu, logvar) ## z = mu + eps*sigma 
        z,adjMat=self.decode(z,adjMat,s1,s2,maxNodes)
        return z, adjMat, mu, logvar,l1,l2 

