from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import pandas as pd
import dlib

import os
from PIL import Image
import pandas as pd
import dlib


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NEED_TRAIN_VGG = 1
DATA_DIR = 'training_data'
TRAIN = 'train'
VAL = 'validation'

data_transforms = {
    TRAIN: transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
 }

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(DATA_DIR, x), 
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=4
    )
    for x in [TRAIN, VAL]
}
imsize = 224
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
CASME2_FILE = 'casme2_img.xls'
SAMM_FILE = 'samm_img_win.xls'

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
         coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def image_loader( image_path ):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)

    return image


def get_micro_expression_VGG16 ( model_file_name ):
    model_load = torch.load(model_file_name)
    

    vgg16 = models.vgg16_bn() 
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, 2)]) # Add our layer with 2  outputs
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    vgg16.load_state_dict(model_load)

    return vgg16

def intersection(L1, L2):
    return list(set(L1) & set(L2))

def prepare_CASME2_train():

    # currently using CASME2 for training, subject 1 for evaluation
    XLS = pd.read_excel(CASME2_FILE)
    num_img = XLS['ID'].shape[0]

    train_index_subject =  XLS.index[XLS['subject'] != 1  ].tolist()
    micro_index = XLS.index[XLS['micro'] > 0  ].tolist()
    # process MICRO-FRAMES
    micro_index_subject = intersection(train_index_subject , micro_index)
    len_micro = len(micro_index_subject)
    dst_folder = '/home/thtran/Project/ME/vgg_lstm/data/train/micro'
    for ind in micro_index_subject:
        src_img = XLS['location'][ind]
        print(src_img)
        shutil.copy2(src_img,dst_folder)


    # # process NEUTRAL-FRAMES
    # print(" PROCESSING NEUTRAL IMAGES ")
    # dst_folder = '/home/thtran/Project/ME/vgg_lstm/data/train/neutral'
    # train_index_subject =  XLS.index[XLS['subject'] != 1  ].tolist()
    # neutral_index = XLS.index[XLS['micro'] == 0  ].tolist()
    # neutral_index_subject = intersection(train_index_subject , neutral_index)

    
    # chosen_neutral_index = sample(neutral_index_subject, len_micro )
    # for ind in chosen_neutral_index:
    #   src_img = XLS['location'][ind]
    #   print(src_img)
    #   shutil.copy2(src_img,dst_folder)
def fine_tune_vgg16_pytorch():

    
    # add images of TRAINIG , VALIDATION
    dataset_sizes = {x: len(image_datasets[x]) for x in [ TRAIN, VAL]}

    for x in [TRAIN , VAL ]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x))
    
    print("Classes: ")

    class_names = image_datasets[TRAIN].classes
    print(image_datasets[TRAIN].classes)
    vgg16 = get_micro_expression_VGG16('Models/VGG16_ME.pt')

    # vgg16 = models.vgg16_bn()
    # vgg16.load_state_dict(torch.load("vgg16_bn-6c64b313.pth"))

    # print(vgg16.classifier[6].out_features) 

    # for param in vgg16.features.parameters():
    #     param.require_grad = True

    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1] # Remove last layer
    # features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
    # vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
    # print(vgg16)

    
    criterion = nn.CrossEntropyLoss()
    #optimizer_ft = optim.Adam(vgg16.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer_sgd = optim.SGD(vgg16.parameters(), lr=0.0001 , momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_sgd, step_size=7, gamma=0.1)

    avg_loss = 0
    avg_acc = 0

    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[VAL])

    for epoch  in range(10):
        print("Epoch {}/{}".format(epoch,10))
        vgg16.train(True)
        loss_train = 0
        acc_train = 0
        for i, data in enumerate(dataloaders[TRAIN]):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer_sgd.zero_grad() 
            outputs = vgg16(inputs)
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_sgd.step()
            
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            print(" Loss Train - Acc Train = ",loss_train," ", acc_train.item() / 5163.0 )
            del inputs, labels, outputs, preds
            
        avg_loss = loss_train * 2 / dataset_sizes[TRAIN]
        avg_acc = acc_train * 2 / dataset_sizes[TRAIN]

        print(" AVG LOSS :  ",avg_loss)
        print("  AVG ACC : ",avg_acc.item())

        vgg16.eval()
        loss_val = 0
        acc_val = 0
        for i, data in enumerate(dataloaders[VAL]):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            #optimizer_ft.zero_grad() 
            outputs = vgg16(inputs)
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            acc_val += torch.sum(preds == labels.data)
            loss_val += loss.item()

            print(" Loss Val - Acc Val = ",loss_val," ", acc_val.item() / 201.0)
            del inputs, labels, outputs, preds
        vgg16.eval()
    torch.save(vgg16.state_dict(), 'VGG16_ME_20200219.pt')

def main():

    print(" Start to Finetune Model ")
    fine_tune_vgg16_pytorch()
    print(" Ending Finetuning model ")
    


if __name__ == '__main__':
    main()  