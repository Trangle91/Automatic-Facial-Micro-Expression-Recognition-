import os
import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, GlobalAveragePooling2D, Activation, BatchNormalization, Dropout
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import backend as K
from prepare_data import SDI_MER
import pickle

TRAIN_FOLDER = './saam_data/train'
VAL_FOLDER = './saam_data/validation'

RAW_DATA = './SAAM_train_data'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
TRAIN_DATA = './saam_data'
L_Res = []

def create_VGG16_model ():
    
    #Load the VGG model
    image_size = 224
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    model = Sequential()
 
    # Add the vgg convolutional base model
    model.add(vgg_conv)
    

     
    # Add new layers
    model.add(Flatten())
    # model.add(Dense(1024, activation='sigmoid'))
    # model.add(Dropout(0.4))
    model.add(Dense(4, activation='softmax'))
    for layer in model.layers:

        print(layer, layer.trainable)
    
    #adam_model = Adam(lr=0.0001, beta_1=0.9, beta_2=0.98, amsgrad=False)
    sgd_model = SGD(lr=0.0001, decay=1e-2, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer= sgd_model ,metrics=['accuracy'])
    
    return model

def create_ResNet50_model():
    image_size = 224
    model = Sequential()

    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    sgd_model = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd_model ,metrics=['accuracy'])
    return model
    
def create_simple_model():
    image_size = 224
    
    if K.image_data_format() == 'channels_first': 
        input_shape = (3, image_size, image_size) 
    else: 
        input_shape = (image_size, image_size, 3) 
    
    model = Sequential() 
    model.add(Conv2D(32, (3, 3), input_shape = input_shape)) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
      
    model.add(Conv2D(32, (3, 3))) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
      
    model.add(Conv2D(64, (3, 3))) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
      
    model.add(Flatten()) 
    model.add(Dense(64)) 
    model.add(Activation('relu')) 
    model.add(Dense(8)) 
    model.add(Activation('softmax')) 
    sgd_model = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd_model ,metrics=['accuracy'])
    return model

def train_model():
    L_Res = []
    train_imagedata  = ImageDataGenerator(rescale = 1. /255)
    val_imagedata = ImageDataGenerator(rescale=1./255)
    training_set = train_imagedata.flow_from_directory(TRAIN_FOLDER , target_size=(224, 224), batch_size=14 , class_mode='categorical')
    val_set = val_imagedata.flow_from_directory(VAL_FOLDER  , target_size=(224, 224), batch_size=14 , class_mode='categorical')
    
    trainning_model = create_VGG16_model()

    res = trainning_model.fit_generator(training_set ,  steps_per_epoch = 50 , epochs = 300 , validation_data=val_set, validation_steps=10 )
    L_Res.append(res)

    return 1
    
def main():
    dat = SDI_MER(RAW_DATA ,TRAIN_DATA )
    
    for i in range(5):
        dat.process_one_fold()
        print(' Start to train !! ')
        train_model()

    with open('L_Res_VGG', 'wb') as fp:
        pickle.dump(L_Res, fp)
    return 0

if __name__ == '__main__':
    main()  
