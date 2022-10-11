#%% 
# #
import pandas as pd
import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset
from clean_images import CleanImages
import pickle
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

le = LabelEncoder()
class FbMarketplaceDataset():
    def __init__(self, data, datapath:str, image_size:int,  decoder:dict = None,
                transforms:transforms = None):
        super().__init__()
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
            decoder = filename
     
        ''' 
        Runs Encoder and Decoder
        '''
        pkl_file = open(decoder, 'rb')
        le_category = pickle.load(pkl_file)
        pkl_file.close()
        self.decoder = lambda x:le_category.inverse_transform(x)
        self.encoder = lambda x: le_category.transform(x)

        ''' 
        Runs Transformer and Sets Default Transformer values
        '''

        if transforms == None:
            self.transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.4217, 0.3923, 0.3633],
                                                                std=[0.3117, 0.2967, 0.2931])])
                                
    def __getitem__(self, index:int):
        label = self.labels[index]
        encoded_label = self.encoder(label)
        label_array = np.asarray(label)
        decoded_text = self.decode(label_array)
        label = torch.as_tensor(label)
        image_id = self.images_id[index]
        os.chdir(f'{self.datapath}/clean_images')
        image = Image.open(f'clean_{image_id}.jpg')
        image = self.transform(image)
        image.close()

    def __len__(self):
        return(len(self.data.image_id))
        
if __name__ == '__main__':
    datapath = '/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/'
    data = pd.read_pickle('clean_image_array.pkl')
    decoder = None
    image_size = 100
    new_fb = FbMarketplaceDataset(data, datapath, image_size)
  

# %%
