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
from prepare_data import SDI_Cross
import pickle


WORKING_FOLDER = './raw_cross_data/'
TRAIN_FOLDER = os.path.join(WORKING_FOLDER,'train')
VAL_FOLDER = os.path.join(WORKING_FOLDER,'validation')

RAW_TRAIN_DATA = './SAAM_train_data'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
RAW_VAL_DATA = './CASMEII_train_data'
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
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    
    #adam_model = Adam(lr=0.0001, beta_1=0.9, beta_2=0.98, amsgrad=False)
    sgd_model = SGD(lr=0.0001, decay=1e-1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer= sgd_model ,metrics=['accuracy'])
    
    return model

def create_ResNet50_model():
    image_size = 224
    model = Sequential()

    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4, activation='softmax'))
    sgd_model = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
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


    
def train_cross_database_model():
    L_Res = []
    train_imagedata  = ImageDataGenerator(rescale = 1. /255)
    val_imagedata = ImageDataGenerator(rescale=1./255)
    training_set = train_imagedata.flow_from_directory(TRAIN_FOLDER , target_size=(224, 224), batch_size=8 , class_mode='categorical')
    val_set = val_imagedata.flow_from_directory(VAL_FOLDER  , target_size=(224, 224), batch_size=8 , class_mode='categorical')
    trainning_model = create_ResNet50_model()

    res = trainning_model.fit_generator(training_set ,  steps_per_epoch = 40 , epochs = 300 , validation_data=val_set, validation_steps=10 )
    L_Res.append(res)
    with open('L_Res_ResNet50', 'wb') as fp:
        pickle.dump(L_Res, fp)

    return 1
    
def main():

    dat = SDI_Cross(  './CASMEII_train_data' , './SAAM_train_data' , './raw_cross_data/' )
    

    dat.process_cross_db()
    print(' Start to train !! ')
    train_cross_database_model()

    
    return 0

if __name__ == '__main__':
    main()  
