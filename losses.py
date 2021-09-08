from __future__ import print_function
# import argparse
import torch
import torch.nn.functional as F
# import numpy as np
import util
###################################################################################
def MSE(output,target):  
    return F.mse_loss(output, target,reduce = None) 
####################################################################
def Cos(output,target):
    Cosdis =  F.cosine_similarity(output.reshape(output.size(0),-1),target.reshape(output.size(0),-1),dim=-1)
    return 1.0 - Cosdis.mean()
####################################################################
def L21(output,target): 
    loss = torch.pow(output - target,2).mean(dim=2).sqrt().mean()  
    return loss
####################################################################
def L1Cos(output,target): 
    loss = torch.abs(output - target).mean(dim=2) 
    return loss
####################################################################
def L1(output,target):  
    return F.l1_loss(output,target,reduce = None)
####################################################################
def histNorm(output,target):
    out = util.seq2img(output)
    tgt = util.seq2img(target) 

    outA = F.normalize(out,dim=1)
    tgtA = F.normalize(tgt,dim=1)

    outM = out.norm(dim=1)
    tgtM = tgt.norm(dim=1)

    return F.l1_loss(outA,tgtA) + F.l1_loss(outM,tgtM)
####################################################################
def L1_Norm(output,target,beta = 1.0): 
    Mout  = output - output.mean(dim=1,keepdim=True)
    Mtgt  = target - target.mean(dim=1,keepdim=True) 
    Sout  = output.norm(dim=1,keepdim=True) 
    Stgt  = target.norm(dim=1,keepdim=True)     
    Normout  = Mout/(Sout + beta)
    Normtgt  = Mtgt/(Stgt + beta)   
    return F.l1_loss(Normout,Normtgt)


