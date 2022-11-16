#%% 
# 
import pandas as pd
import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset
import pickle
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

le = LabelEncoder()
le = LabelEncoder()
def repeat_channel(x):
            return x.repeat(3, 1, 1)

class FbMarketImageDataset(Dataset):
    def __init__(self, image_size:int=100,  decoder:dict = None,
                transformer:transforms = None):
        super().__init__()
        self.datapath = '/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking'
        data = pd.read_pickle(f'{self.datapath}/data/clean_image_array.pkl')
        self.data= data
        self.image_id = self.data['image_id'].to_list()
        self.labels= self.data['main_category'].to_list()
        self.transformer = transformer
        

        ''' 
        Sets up Default Decoder
        '''
        if decoder == None:
            le = LabelEncoder()
            self.data['cat_codes'] = le.fit_transform(data['main_category'].astype('category'))
            filename = 'image_decoder.pkl'
            output = open(filename,'wb')
            pickle.dump(le, output)
            output.close()
            
     
        ''' 
        Runs Encoder and Decoder
        '''
        decoder = filename
        pkl_file = open(decoder, 'rb')
        le_category = pickle.load(pkl_file)
        pkl_file.close()
        self.encode = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decode = {x: y for (x, y) in enumerate(set(self.labels))}

        ''' 
        Runs Transformer and Sets Default Transformer values
        '''

      
        if self.transformer is None:
            self.transformer = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]) # is this right?
            ])

            self.transformer_Gray = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Lambda(repeat_channel),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

                                
    def __getitem__(self, index:int):
        label = self.labels[index]
        label = self.encode[label]
        label = torch.as_tensor(label)
        os.chdir(f'{self.datapath}/clean_images')
        image = Image.open(f'clean_{self.image_id[index]}.jpg')
        if image.mode != 'RGB':
            feature = self.transformer_Gray(image)
        else:
            feature = self.transformer(image)

        return feature, label

    def __len__(self):
        return(len(self.data.image_id))
    
    def get_category(self, label:int):
        category = self.decode[label]
        print(category)
        
if __name__ == '__main__':
    new_fb = FbMarketImageDataset()
    print(new_fb[9])
    new_fb.get_category(9)

    #Create Data Loader and Test
    image_loader = DataLoader(new_fb, batch_size = 32, shuffle = True, num_workers = 1)
    for i, (features,labels) in enumerate(image_loader):
        print(features)
        print(labels)
        print(features.size())
        if i == 0:
            break
