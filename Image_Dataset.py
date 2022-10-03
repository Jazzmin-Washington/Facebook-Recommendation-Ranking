#%% 
# #
import pandas as pd
import os
import json
from torch.utils.data import Dataset
from clean_images import CleanImages
import pickle
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

le = LabelEncoder()
class FbMarketplaceDataset():
    def __init__(self, data, datapath:str, image_size:int,  decoder:dict = None,
                transforms:transforms = None):
        super().__init__()
        self.image_id = self.data['image_id'].to_list()
        self.labels= self.data['main_category'].to_list()
       
        

        ''' Sets up Default Decoder'''
        if decoder == None:
            le = LabelEncoder()
            self.data['cat_codes'] = le.fit_transform(data['main_category'].astype('category'))
    
            filename = 'image_decoder.pkl'
            output = open(filename,'wb')
            pickle.dump(le, output)
            output.close()
            decoder = filename
     
         ''' Runs Encoder and Decoder'''
        pkl_file = open(decoder, 'rb')
        le_category = pickle.load(pkl_file)
        pkl_file.close()
        decoder = lambda x:le_category.inverse_transform(x)
        encoder = le_category.transform(self.labels)

        ''' Runs Transformer and Sets Default Transformer values'''
        if transforms == None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224, 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4217, 0.3923, 0.3633],
                                        std=[0.3117, 0.2967, 0.2931])])

    def __getitem__(self, index:int):
        label = self.labels[index]
        label_array = np.asarray(label)
        decoded_text = decode(label_array)
        label = torch.as_tensor(label)
        image = Image.open(f'{self.datapath}/clean_images/clean_{self.image_id[index]}.jpg')
        image = self.transform(image)
        image.close()
        return image, label, decoded_text

    def __len__(self):
        return(len(self.data.image_id))
        
if __name__ == '__main__':
    datapath = '/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/'
    data = pd.read_pickle('clean_image_array.pkl')
    decoder = None
    image_size = 100
    new_fb = FbMarketplaceDataset(data, datapath, image_size)
  

# %%
