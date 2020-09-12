import os
import numpy as np
import cv2
import shutil
from os import listdir
import dlib

RAW_DATA = './raw_data'
def split_data_random():
    return 1

def split_data_LOSO(folder_address):

    cross_list = []
    subject_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]



    return 1

def lable_name (img_name):
    
    first_name = img_name.split('_')[0]
    
    if ('Ang' in first_name  or 'Neg' in first_name or  'Dis' in first_name):
        return str(0)
    if ('Fear' in first_name  or 'Other' in first_name or  'Sad' in first_name):
        return str(1)
        
    if ('Happ' in first_name  or 'Posi' in first_name ):
        return str(2) 
    
    if ('Sur' in first_name ):
        return str(3) 
    
    return str(1)
    
def process_img(folder_in , img_name  , folder_out , aug = 0 ):
    img = cv2.imread(os.path.join(folder_in ,img_name )  )
    print(img_name)

    (h,w,c) = img.shape

    temp = np.zeros([h,w,c])

    temp[0:45,:,:] = 1.0
    temp[0:140,112-20:112+21,:] = 1.0
    temp[140:200,112-60:112+61,:] = 1.0

    new_img = img * temp
    new_img = new_img.astype(int)


    out_img_path = os.path.join(folder_out ,lable_name(img_name) + '_' +  img_name)

    cv2.imwrite(out_img_path , new_img  )

    # fplit image 
    if (aug == 1):
        fplip_img = cv2.flip(img, 1)
        new_img = fplip_img * temp
        new_img = new_img.astype(int)
        out_img_path = os.path.join(folder_out , lable_name(img_name) +  '_flip_' + img_name)
        cv2.imwrite(out_img_path , new_img  )

    return 1


class SDI_MER():
    def __init__(self , raw_data_folder , working_folder , facial_region = 0, test_ratio = 0.1 , num_cross = 10 ):
        self.raw_data_folder = raw_data_folder

        if (os.path.exists(working_folder)==False):
            os.mkdir(working_folder)
        self.train_folder = os.path.join(working_folder,'train')
        self.validation_folder = os.path.join(working_folder,'validation')
        self.create_folder(self.train_folder )
        self.create_folder(self.validation_folder )

        self.using_face_region = facial_region
        self.test_ratio = test_ratio
        self.num_cross = num_cross

    def process_one_fold(self  ):
        list_folder = listdir(self.raw_data_folder)
        print(list_folder)
        for fo in list_folder:
            if fo[0] != '.':
                class_folder = os.path.join(self.raw_data_folder , fo)
                list_img = listdir(class_folder)

                n_img = len(list_img)
                np_permu = np.random.permutation(n_img)
                n_train = int(n_img * (1 - self.test_ratio))
                train_idx = np_permu[0:n_train ]
                val_idx = np_permu[n_train:n_img]
                
                for i in range(n_train):
                    idx = np_permu[i]
                    img_name = list_img[idx]
                    if (img_name[0] != '.'):
                        process_img(class_folder , img_name, self.train_folder , 1)

                for i in range(n_train,n_img):
                    idx = np_permu[i]
                    img_name = list_img[idx]
                    if (img_name[0] != '.'):
                        process_img(class_folder , img_name, self.validation_folder , 0)




    def copy_img_to_folder(self, XLS , img_id , folder_address):
    
        location = XLS['location']
        for id in img_id:
            src_img_path = location[id]
            shutil.copy(src_img_path , folder_address )
    
    def create_folder (self  , input_folder ):
        if (os.path.exists(input_folder)==False):
            os.mkdir(input_folder)
        else:
            shutil.rmtree(input_folder)
            os.mkdir(input_folder)
            
class SDI_Cross():
    def __init__(self , raw_train_folder , raw_val_folder , working_folder  ):
        self.raw_train_folder = raw_train_folder
        self.raw_val_folder = raw_val_folder
        if (os.path.exists(working_folder)==False):
            os.mkdir(working_folder)
        self.train_folder = os.path.join(working_folder,'train')
        self.validation_folder = os.path.join(working_folder,'validation')
        self.create_folder(self.train_folder )
        self.create_folder(self.validation_folder )
        
    def process_cross_db( self ):
    
        # process the training  data
        list_folder = listdir(self.raw_train_folder)
        print(list_folder)
        for fo in list_folder:
            if fo[0] != '.':
                class_folder = os.path.join(self.raw_train_folder , fo)
                list_img = listdir(class_folder)
                
                train_class_folder = os.path.join(self.train_folder,fo)
                self.create_folder(train_class_folder)
                
                n_img = len(list_img)
                for i in range(n_img):
                    img_name = list_img[i]
                    if (img_name[0] != '.'):
                        process_img(class_folder , img_name, train_class_folder , 1)
                
        # process the validation data
        list_folder = listdir(self.raw_val_folder)
        print(list_folder)
        for fo in list_folder:
            if fo[0] != '.':
                class_folder = os.path.join(self.raw_val_folder , fo)
                list_img = listdir(class_folder)
                
                val_class_folder = os.path.join(self.validation_folder,fo)
                self.create_folder(val_class_folder)
                
                n_img = len(list_img)
                for i in range(n_img):
                    img_name = list_img[i]
                    if (img_name[0] != '.'):
                        process_img(class_folder , img_name, val_class_folder , 0)
        
    
        
    def copy_img_to_folder(self, XLS , img_id , folder_address):
    
        location = XLS['location']
        for id in img_id:
            src_img_path = location[id]
            shutil.copy(src_img_path , folder_address )
    
    def create_folder (self  , input_folder ):
        if (os.path.exists(input_folder)==False):
            os.mkdir(input_folder)
        else:
            shutil.rmtree(input_folder)
            os.mkdir(input_folder)

def main():
    dat = SDI_MER('CASMEII_train_data', 'saam_data')
    
    return 0

if __name__ == '__main__':
    main()
