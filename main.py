
#########################################################################
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from util import seed_torch,load_NTUdataset,img2seq,seq2img,load_uwa3d_dataset,load_ucla_dataset
from model import UnsupervisedNet,SupervisedNet
import numpy as np
import os
from runx.logx import logx
import util
import losses
from tqdm import tqdm
#########################################################################
version = 3
#########################################################################
def train(args, model, device, train_loader, optimizer, epoch):
    model.train() 
    ##################################################################
    with tqdm(enumerate(train_loader)) as t: 
        for batch_idx, (data,labels) in t:
            ######################################################################
            t.set_description(" L1 and Epoch %i"%epoch)
            #print(data.size())
            inputs = util.img2seq(data).to(device)     
            #############################################        
            loss = model(inputs)[0]    
            ###################################################
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step() 
            if batch_idx % 100 == 0:
                metrics = {'loss': loss.item()}
                iteration = epoch * len(train_loader) + batch_idx
                logx.metric('train', metrics, iteration)

def get_features(model, data_loader,device):
    model.eval()
    features,labels= [],[]
    with torch.no_grad():
        for _, (inputs, label) in enumerate(data_loader):
            label  = label.to(device)
            inputs = util.img2seq(inputs).to(device)   
            feature = model(inputs,istraing=False)
            features.append(feature) 
            labels.append(label)
    features = torch.cat(features)
    labels   = torch.cat(labels).long()
    return features,labels
###################################################################################
def test(args, model, device, train_loader,test_loader):
    model.eval()
    features_train,target_train = get_features(model,train_loader,device)
    features_test, target_test  = get_features(model,test_loader,device)
    
    # 
    norm_features_train = F.normalize(features_train,dim = -1)
    norm_features_test  = F.normalize(features_test, dim = -1)
    #
    print(norm_features_train.size())
    
    distmat =  torch.matmul(norm_features_train,norm_features_test.transpose(0,1)) 
    #
    Indx = torch.argmax(distmat,dim=0) 
    correct  = target_test.eq(target_train[Indx]).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


#######################################################################    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Unspervised skeleton')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
    parser.add_argument('--datadir', type=str, default='rawNTU60_50')  #NTU120 
    parser.add_argument('--evaluation', type=str, default='CS',help='evaluation (default: CS)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.001)')
    parser.add_argument('--LR_STEP', type=str, default='80', metavar='LS',help='LR_STEP (default: 100)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')  
    parser.add_argument('--save_dir', type=str, default='Triple')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--logdir', type=str, default='./logdir_120_cs')
    #######################
    parser.add_argument('--gpuid', type=int, default=4,help='useing cuda devices id')
    parser.add_argument('--en_layers', type=int, default=1,help='encoder layers')
    parser.add_argument('--de_layers', type=int, default=1,help='decoder layers')
    #############################################################
    args = parser.parse_args()

    logx.initialize(logdir=args.logdir, coolname=True, tensorboard=False,hparams=vars(args))

    print("%s_%s, gpuid %d:"%(args.datadir,args.evaluation,args.gpuid)) 
    seed_torch(args.seed)
    device = torch.device('cuda',args.gpuid)
    savedir = './checkpoint/%s/%s%s'%(args.datadir,args.evaluation,args.save_dir)    
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    ####################################################################################
    # load dataset
    traindsets = load_NTUdataset(datadir=args.datadir,mode='train',evaluation = args.evaluation)
    #traindsets = load_uwa3d_dataset(mode='train')
    #traindsets = load_ucla_dataset(mode='train')
    train_loader =  torch.utils.data.DataLoader(dataset=traindsets,num_workers=16,
                                                batch_size= args.batch_size,shuffle=True)

    testdsets = load_NTUdataset(datadir=args.datadir,mode='val',evaluation = args.evaluation)
    #testdsets = load_uwa3d_dataset(mode='test',version=version)
    #testdsets = load_ucla_dataset(mode='test')
    test_loader =  torch.utils.data.DataLoader(dataset=testdsets,num_workers=16,
                                                batch_size= args.batch_size,shuffle=False) 
    # testdsets4 = load_uwa3d_dataset(mode='test',version=4)
    # test_loader4 =  torch.utils.data.DataLoader(dataset=testdsets4,num_workers=16,
    #                                             batch_size= args.batch_size,shuffle=False)  
    #                                   
    ####################################################################################
    model = UnsupervisedNet().to(device)
    optimizer  = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.LR_STEP, args.lr_decay_rate)
    best_acc=0.0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        lr_scheduler.step()
        if epoch %1 ==0:
            accuracy = test(args, model, device, train_loader,test_loader)          #test_loader
            logx.msg("epoch:{} \033[0m is {:.2f}".format(epoch,accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                best_acc = accuracy
                torch.save(model, os.path.join(savedir, 'model_best_NTU120_64.pth'))
    print(best_acc)

         

if __name__ == '__main__':
    main()
