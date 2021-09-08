from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from numpy.random import beta
from os.path import join
import numpy as np
import h5py
import pickle
import random
import os
from sklearn.metrics import confusion_matrix
################################################################
def seed_torch(seed=1):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
################################################################
def load_NTUdataset(datadir='rawNTU60_50',mode='train',evaluation='CS'):   #datadir='NTU60'  rawNTU60_50 rawNTU120
    ############################################
    rootdir = '/home/data'
    datadir = join(rootdir,datadir,'xsub' if evaluation== 'CS' else 'xview')
    x = np.load(join(datadir, mode + '_data_joint.npy'))
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], -1)) 
    with open(join(datadir, mode + '_label.pkl'),'rb') as file:
        label = pickle.load(file)[1]
    y = np.vstack(label).reshape(-1)
    ############################################x
    dsets = torch.utils.data.TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()) 
    return dsets 
############################################ 
def load_uwa3d_dataset(mode='train', type='zoomed',version=3):   #version=4
    path = '/home/data/new_data/uwa3d_ver3andver4/'
    if mode == 'train':   
        data = path + mode + '_data.npy'
        label = path + mode + '_label.npy'
        data = np.load(data)
        label = np.load(label)
        dsets = torch.utils.data.TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(label).long())
    elif mode == 'test': 
        data = path + mode + 'version'+ str(version) + '_data.npy'
        label = path + mode + 'version'+ str(version) + '_label.npy'
        data = np.load(data)
        label = np.load(label)
        dsets = torch.utils.data.TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(label).long())
    return dsets

def load_ucla_dataset(mode='train', type='zoomed',version=4):

    path = '/home/data/new_data/ucla_data/'
    if mode == 'train':   
        data = path + mode + '_data.npy'
        label = path + mode + '_label.npy'
        data = np.load(data)
        label = np.load(label)
        dsets = torch.utils.data.TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(label).long())
    elif mode == 'test':
        data = path + mode + '_data.npy'
        label = path + mode + '_label.npy'
        data = np.load(data)
        label = np.load(label)
        dsets = torch.utils.data.TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(label).long())
    return dsets

###################################################################################
def img2seq(img):
    #print(img.size())
    seq = img.permute(0,2,1,3).reshape(img.size(0),img.size(2),-1) if len(img.size())==4 else img
    #print(seq.size())
    return seq
  #########################################################################################################   

def seq2img(seq):
    img = seq.reshape(seq.size(0),seq.size(1),3,-1).permute(0,2,1,3) if len(seq.size())==3 else seq
    # print(img.size())
    return img

##
########################################################################################## 
def recover_Floss(model, device,test_loader,mode='MSE'): 
    model.eval() 
    loss_flow = 0.0
    lossA_recflow = 0.0
    lossD_recflow = 0.0
    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            # target = target.to(device)
            target = data.to(device)
            outputs = model(target)[0]
            flow_target   = target - target.mean(dim=2,keepdim=True)
            flow_targetA  = flow_target.norm(dim=1,keepdim=True)
            flow_targetD  = flow_target/(flow_targetA + 1e-10)
            if mode == 'MSE':
                flow_outputs   = outputs[0] - outputs[0].mean(dim=2,keepdim=True)
                flow_outputsA  = flow_outputs.norm(dim=1,keepdim=True)
                flow_outputsD  = flow_outputs/(flow_outputsA + 1e-10)
            elif mode == 'PRF':
                flow_outputs   = outputs[0]
                flow_outputsA  = flow_outputs.norm(dim=1,keepdim=True)
                flow_outputsD  = flow_outputs/(flow_outputsA + 1e-10)
            elif mode == 'PDF_A':
                flow_outputs   = outputs[0].repeat(1,3,1,1)
                flow_outputsA  = outputs[0]
                flow_outputsD  = flow_outputs
            elif mode == 'PDF_D':
                flow_outputs   = outputs[0]
                flow_outputsA  = outputs[0].norm(dim=1,keepdim=True)
                flow_outputsD  = outputs[0]/(flow_outputsA + 1e-10)
            else:
                flow_outputs   = outputs[0] - outputs[0].mean(dim=2,keepdim=True)
                flow_outputsA  = outputs[0] - outputs[0].mean(dim=2,keepdim=True)
                flow_outputsA  = flow_outputsA.norm(dim=1,keepdim=True)
                flow_outputsD  = outputs[1] - outputs[1].mean(dim=2,keepdim=True)
            
            # flow_outputsD = outputs[1] - outputs[1].mean(dim=2,keepdim=True)
            rec_flow      = flow_outputsA/(flow_outputsD.norm(dim=1,keepdim=True) + 1e-10) * flow_outputsD

            loss_flow     = loss_flow +  target.size(0)*F.mse_loss(rec_flow,flow_target) 
            lossA_recflow = lossA_recflow + target.size(0)*F.mse_loss(flow_outputsA,flow_targetA) 
            lossD_recflow = lossD_recflow + target.size(0)*(1.0 - F.cosine_similarity(flow_outputsD,flow_targetD,dim=1).mean())

    print('Output size is ',outputs[0].size()[1:])
    return loss_flow,lossA_recflow,lossD_recflow

##########################################################################################
if __name__ == '__main__':
    evaluation = 'CS'
    mode = 'PDF_D'
    device = torch.device('cuda',2)
    datadir = '/dev/Data/NTU/LMY'
    testdsets = load_NTUdataset(datadir=datadir,mode='train',evaluation =evaluation)
    test_loader =  torch.utils.data.DataLoader(dataset=testdsets,num_workers=16, batch_size= 128,shuffle=False)  
    modeldir = './checkpoint/%s/%s/model_%d.pth'%(evaluation,mode,100)
    print(modeldir)
    model = torch.load(modeldir,map_location='cuda:2')
    loss,loss_A,loss_D = recover_Floss(model,device,test_loader,mode=mode)
 
    print('loss Flows, is %f loss_A is %f and loss_D is %f'%(loss.item(),loss_A.item(),loss_D.item()))
