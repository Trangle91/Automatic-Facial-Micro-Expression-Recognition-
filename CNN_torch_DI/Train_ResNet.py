from os import path
import torch
import PIL.Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,models
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from collections import OrderedDict
import warnings
import os
import math
import utils
import cv2
import Senet50_model
import Resnet_model
import VGG_model
import vgg_face_dag
from prepare_data import SDI_MER
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

warnings.filterwarnings("ignore")
##################################################################################
class Database(Dataset):
    def __init__(self,root):
        self.Files=[]
        for r, d, f in os.walk(root):
            for file in f:
                if '.jpg' in file:
                    self.Files.append(os.path.join(r, file))
        self.mean_bgr = np.array([93.59396362304688, 104.76238250732422, 129.186279296875])      
    def __len__(self):
        return len(self.Files)
    def __getitem__(self, idx):
        img_name = self.Files[idx]
        head,tail=os.path.split(img_name)
        name=tail.split('_')        
        target=int(name[0])    
        img = cv2.imread(img_name)        
        img = cv2.resize(img,(224,224))
        img=img.astype(np.float32)
        img /=255
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        assert len(img.shape) == 3
        return img,target
####################################################################################
def Acuuracy(predicted,real):
    predicted=predicted.cpu().data.numpy()
    real=real.cpu().numpy().astype(int)
    accuracy=np.zeros(np.size(predicted,1))
    count=np.zeros(np.size(predicted,1))
    for i in range(np.size(real,0)):
        if np.argmax(predicted[i,:])==real[i]:
            accuracy[real[i]]+=1
        count[real[i]]+=1
    return accuracy,count   
##########################################################################
def train_epoch(model, train_loader,criterion, optimizer,epoch):
    model.train()
    global_loss=[]  
    for i, (images, target) in enumerate(train_loader):
        if torch.cuda.is_available():
           images, target=images.cuda(), target.cuda()
        images = Variable(images)
        optimizer.zero_grad()
        emotions= model(images)
        loss= criterion(emotions, target)     
        loss.backward()
        optimizer.step()
        global_loss=np.append(global_loss,loss.data.item())
        if i % 100 == 0:
          print('Epoch: {} \t Train :[{}/{}] \t loss ={:.6f}'.format(epoch,i * len(images), len(train_loader.dataset),np.mean(global_loss)))
    return np.mean(global_loss)
###################################################################################
def validation_epoch(model, dev_loader,criterion):
    model.eval()
    global_loss=[]
    accuracy=np.zeros(4)
    count=np.zeros(4)
    for i, (images, target) in enumerate(dev_loader):
        if torch.cuda.is_available():
           images, target=images.cuda(), target.cuda()
        images = Variable(images)
        emotions = model(images)
        loss= criterion(emotions, target)
        global_loss=np.append(global_loss,loss.cpu().data.item())
        acc,cpt=Acuuracy(emotions,target)

        accuracy+=acc
        count+=cpt
    return np.mean(global_loss),(accuracy/count)*100
####################################################################################
def main():
    
    L_acc = []
    folder_model = './Models/Resnet_50'
    
    best_model = os.path.join(folder_model , 'Best_Model.pth')
    last_model = os.path.join(folder_model , 'Last_Model.pth')
    
    for i in range(10):
        os.mkdir(folder_model)
    
        dat = SDI_MER('CASMEII_train_data' , 'data_resnet' )
        dat.process_one_fold()
        
        
        start_epoch,max_epoches,best_acc=0,200,0
        Training_loss=[]
        Validation_loss=[]
        Accuracy = []
        model=Resnet_model.resnet50()
        criterion=nn.CrossEntropyLoss()
        if torch.cuda.is_available():
           model=model.cuda()      
           criterion=criterion.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,weight_decay=0.0001,momentum=0.9)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)
        ###################### Resume checkpoint  ##################################
        try:
            state=torch.load(last_model) 
            state_dict=state['state_dict']
            model.load_state_dict(state_dict)
            optim_dict=state['optimizer']
            optimizer.load_state_dict(optim_dict)               
            Training_loss=state['Training_loss']
            Validation_loss=state['Validation_loss']
            Accuracy=state['Accuracy']
            start_epoch=state['epoch']      
            best_acc=state['best_acc']
        except:
            print('There is no model available')   
        ################################################################################    
        train_data=Database('./data1/train/')
        dev_data=Database('./data1/validation/')
        train_loader = DataLoader(train_data,
                             batch_size=10,
                             shuffle=True,
                             num_workers=4)
        dev_loader = DataLoader(dev_data,
                             batch_size=10,
                             shuffle=False,
                             num_workers=4)
        for epoch in range(start_epoch,max_epoches):
            train_loss=train_epoch(model, train_loader,criterion, optimizer,epoch)
            validation_loss,accuracy=validation_epoch(model, dev_loader,criterion)
            Validation_loss=np.append(Validation_loss,validation_loss)
            Training_loss=np.append(Training_loss,train_loss)
            Accuracy=np.append(Accuracy,accuracy)
            print('Epoch: {} \t Train_loss: {:.3f} \t Val_Loss: {:.3f} \t Acc: {}'.format(epoch,train_loss,validation_loss,np.mean(accuracy)))
            print(accuracy)
            if (np.mean(accuracy)>best_acc):
                state={
                    'state_dict': model.state_dict(),
                    'best_acc':accuracy,
                }
                torch.save(state,best_model)
                best_acc=np.mean(accuracy)
            state={
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'best_acc':best_acc,
                    'Training_loss':Training_loss,
                    'Validation_loss':Validation_loss,
                    'Accuracy':Accuracy,
                    'epoch':epoch,
                }
            torch.save(state,last_model)
        sta = torch.load(best_model)
        
        np_acc = sta['best_acc']
        shutil.rmtree(folder_model)

        L_acc.append(np.mean(np_acc))
    print(L_acc)
if __name__ == '__main__':
    main()
