import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_dense_batch, to_dense_adj
#import neuralnet_pytorch
from wassy import SinkhornDistance



def loss_function(whole,labels,nodeCount, lengs,refMat,mu,sig):
    #whole=to_dense_batch(labels, lengs, fill_value=0, max_num_nodes=1000)[0].cuda()

    layers=torch.nn.BCEWithLogitsLoss()
    loss2 = torch.nn.SmoothL1Loss()
    mae=torch.nn.L1Loss()
    loss=torch.nn.MSELoss()
    #sinkhorn=SinkhornDistance(eps=0.1, max_iter=100)
    """
    emp=whole[0][:nodeCount[0]]
    for k in range(1,whole.shape[0]):
      emp=torch.cat((emp,whole[k][:nodeCount[k]]),dim=0)
    
    empL=labels[0][:nodeCount[0]]
    for kk in range(1,labels.shape[0]):
      empL=torch.cat((empL,labels[kk][:nodeCount[kk]]),dim=0)

    emp=emp.cuda()
    cost1=loss(emp[:,0],empL[:,0])+loss(emp[:,1],empL[:,1])+loss(emp[:,1],empL[:,1])
    """
    #cost1 =  neuralnet_pytorch.metrics.chamfer_loss(emp.cuda(), labels, reduce='mean', c_code=True)#lossMSE(r1, labels)
    #cost1=sinkhorn(r1, labels)
    #cost1=neuralnet_pytorch.metrics.emd_loss(r1, whole, reduce='mean', sinkhorn=True)

    cost1=loss(whole[:,:,0],labels[:,:,0]) + loss(whole[:,:,1],labels[:,:,1])+ loss(whole[:,:,2],labels[:,:,2])
    layloss=loss(whole[:,:,3],labels[:,:,3])
    #KLD = -0.5 *(torch.mean(torch.sum(1 + sig - mu.pow(2) - sig.exp(), 1)))
    #KLD = -0.5 * torch.sum(1 + sig - mu.pow(2) - sig.exp())
    KLD = -0.5 * torch.mean(1 + sig - mu.pow(2) - sig.exp())
    
    return cost1  + KLD +layloss
