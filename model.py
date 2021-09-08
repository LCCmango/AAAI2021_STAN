from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import util 
from losses import * 
import torchvision.models as models 
import numpy as np
import torch.optim as optim 
####################################################################################################    
import torch
import torch.nn as nn

Indx = [0,1,2,4,8,12,16,20]
# Indx = [0,1,2,4,8]
####################################################################################################  
class Normlayer(nn.Module):
    def __init__(self,Seq = 50):
        super(Normlayer, self).__init__()  
        self.gama = nn.Parameter(torch.eye(Seq).unsqueeze(dim=0))  
    def forward(self, x): 

        Norm = x.mean(dim=2,keepdim=True)
        x    =  x / ( 1e-6 + Norm)
        Mean = torch.mean(x,dim=1,keepdim=True) 
        x    = x - Mean 
        return x        
####################################################################################################                               
class Encoder(nn.Module):
    def __init__(self,input_dim = 150,features_dim=256,num_layers = 1,bidirectional=False):  
        super(Encoder, self).__init__()
        #######################################################################################
        self.encoder   = nn.GRU(input_size = input_dim,hidden_size = features_dim,
                                num_layers = num_layers,batch_first= True,bidirectional=bidirectional)   
        self.BNlayer   = nn.BatchNorm1d(features_dim) 
        self.std =  nn.Linear(features_dim, features_dim)

    ################################################################################
    def forward(self, seq):   
        _,h0   =  self.encoder(seq) 
        features  = F.normalize(self.BNlayer(h0.mean(dim=0)),dim=1)           
        return features
        #return  self.BNlayer(h0.mean(dim=0))
        

################################################################################                             
class Decoder(nn.Module):
    def __init__(self,input_dim = 256,output_dim=150,seqlen = 50,num_layers=1): #150
        super(Decoder, self).__init__()
        self.seqlen       = seqlen  
        self.Temporal     = nn.Sequential(nn.Linear(input_dim,seqlen),nn.ReLU(),nn.BatchNorm1d(seqlen),
                                        nn.Linear(seqlen,seqlen),nn.Sigmoid())
        self.decoder      = nn.GRU(input_size = input_dim,hidden_size = input_dim, 
                                num_layers = num_layers,batch_first= True,bidirectional=False) 
        self.dense_layers = nn.Linear(input_dim,output_dim) 
    ################################################################################
    def forward(self, features):     
        Temporal = self.Temporal(features).reshape(-1,self.seqlen,1)
        fstyles  = Temporal * features.unsqueeze(dim=1)
        seq      = self.dense_layers(self.decoder(fstyles)[0]) 
        return seq
                            

###############################################################################    
class UnsupervisedNet(nn.Module):
    def __init__(self,input_dim = 150,seqlen = 50,features_dim = 256,en_layers = 1,de_layers = 1): #input_dim = 150
        super(UnsupervisedNet, self).__init__()  

        self.encoder  = Encoder(input_dim = input_dim,features_dim = features_dim,num_layers = en_layers)  
        self.decoder  = Decoder(input_dim = features_dim,output_dim = input_dim,num_layers = de_layers,seqlen = seqlen)  
        self.beta     = 1.0   
        self.bn= nn.BatchNorm1d(50, affine=False)
        self.ln=nn.LayerNorm([50,150])
        self.inn = nn.InstanceNorm1d(50, affine=False)
      
    def EnergyEq(self,x): 
        #
        Energy  = x.norm(dim=2)
        CumEn   = Energy.cumsum(dim=1)  
        Prob    = CumEn/CumEn[:,-1].unsqueeze(-1)  
        mapping = ((x.size(1) -1) * Prob).long()
        y = torch.stack([x[i,mapping[i,:],:] for i in range(x.size(0))])                  
        return y


        
    def STAN(self,x):

        Mean = x.mean(dim=1,keepdim=True)
        x    = x - Mean
        
        Norm = x.std(dim=2,keepdim=True)
        x = x/(1e-6 + Norm)
        

        return x       
        

    def loss_fun(self,output,target):  

        out = self.STAN(output)
        tgt = self.STAN(target)
        loss = MSE(out,tgt)
        return loss
      

    def forward(self, input,istraing=True): 
        if istraing: 
            # inputs   = input[:,torch.randperm(input.size(1))<torch.randint(20,input.size(1),(1,)),:]
            features = self.encoder(input)
            output  = self.decoder(features)  
            loss = self.loss_fun(output,input)  
            return loss,output,features
        else:
            features = self.encoder(input)
            return features 
################################################################################

class SupervisedNet(nn.Module):
    def __init__(self,input_dim = 256,classnum = 60):
        super(SupervisedNet, self).__init__()  
        self.cl  =  nn.Linear(input_dim,classnum)
    def forward(self, features): 
        return self.cl(features) 
# ####################################################################################################                               
class HARNet(nn.Module):
    def __init__(self,input_dim = 60,features_dim=128,classnum=60,bidirectional=False):
        super(HARNet, self).__init__()
        #######################################################################################
        self.encoder11   = nn.GRU(input_size = input_dim,hidden_size = features_dim,
                                num_layers = 1,batch_first= True,bidirectional=bidirectional) 
        
        self.encoder21   = nn.GRU(input_size = features_dim,hidden_size = features_dim,
                                num_layers = 1,batch_first= True,bidirectional=bidirectional) 

        self.encoder31   = nn.GRU(input_size = features_dim,hidden_size = features_dim,
                                num_layers = 1,batch_first= True,bidirectional=bidirectional) 
        self.features  = nn.Sequential(nn.Linear(features_dim,classnum),nn.BatchNorm1d(classnum))  
    ################################################################################
        self.beta  = 1e-6
        self.GDN1  = Normlayer()
        self.GDN2  = Normlayer()



    def forward(self, x,target):


        x11,_  =  self.encoder11(x)


        x21,_  =  self.encoder21(x11) 
 

        _,h  =  self.encoder31(x21)  

        features = self.features(h.mean(dim=0))

        loss  = F.cross_entropy(features,target)
        return  features,loss

##########################################################################################
if __name__ == '__main__':
    inputs = torch.rand(16,3,50,50) 
    model = AENet()  
    outputs,features,loss = model(inputs)
    print('features size',features[0].size()) 
    print('features size',features[1].size()) 
    print('outputs size',outputs[0].size())




