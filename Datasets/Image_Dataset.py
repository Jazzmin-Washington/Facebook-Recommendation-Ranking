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
class FbMarketImageDataset(Dataset):
    def __init__(self, image_size:int=100,  decoder:dict = None,
                transformer:transforms = None):
        super().__init__()
        self.datapath = '/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking'
        data = pd.read_pickle(f'{self.datapath}/data/clean_image_array.pkl')
        self.data= data
        self.image_id = self.data['image_id'].to_list()
        self.labels= self.data['main_category'].to_list()
        

       
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
        self.decode = lambda x:le_category.inverse_transform([x])
        self.encode = lambda x: le_category.transform([x])

        ''' 
        Runs Transformer and Sets Default Transformer values
        '''

        if transformer == None:
            self.transform = transforms.Compose([transforms.Resize(96),
                                            transforms.CenterCrop(96), 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.4217, 0.3923, 0.3633],
                                                                std=[0.3117, 0.2967, 0.2931])])
                                
    def __getitem__(self, index:int):
        label = self.labels[index]
        label = self.encode(label)
        label = torch.tensor([label]).to(dtype=torch.long)
        image_id = self.image_id[index]
        os.chdir(f'{self.datapath}/clean_images')
        image = Image.open(f'clean_{image_id}.jpg').convert('RGB')
        feature = self.transform(image)
        image.close()
        print(index)
        return feature, label

    def __len__(self):
        return(len(self.data.image_id))
    
    def get_category(self, label:int):
        category = self.decode(label)
        print(category)
        
if __name__ == '__main__':
    new_fb = FbMarketImageDataset()
    print(new_fb[9])
    new_fb.get_category(9)

    #Create Data Loader and Test
    image_loader = DataLoader(new_fb, batch_size = 32, shuffle = True)
    for batch in image_loader:
        print(batch)
        features, labels = batch
        print(features.shape)
        print(labels.shape)

# %%
