#%%
# Create and Image Classifier
import os
import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from alive_progress import alive_bar
from pandas import to_pickle
from PIL import Image
from torchvision.transforms import ToTensor
from clean_images import CleanImages


class CreateImageProcessor():
    def __init__(self, products):
        self.images_w_products = products
        self.image_array = []
        self.image_id = []
        self.image_info = pd.DataFrame()
        self.datapath = '/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking'
        self.clean_image_save = f'/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/clean_images'
    
        
    def create_save_folder(self):
        if not os.path.exists(self.clean_image_save):
            os.makedirs (self.clean_image_save , exist_ok=True)
        
    def compile_images(self):
        self.images_w_products.drop('Unnamed: 0', inplace=True, axis=1)
        self.images = self.images_w_products[['image_id','product_id', 'main_category']]
        print(self.images.info())
        self.image_list = self.images['image_id'].to_list()
    
    
    def resize_images(self):
            cleaner = CleanImages()
            print("Let's begin reformatiing the images...")
            cleaner.run_image_cleaner()
            print(f"Success! Reformatted Images were saved at: {self.clean_image_save} ")

    def images_to_array(self):
        print('Converting Images to Numpy Array...')
        os.chdir(self.clean_image_save)
        print(len(os.listdir()))
        with alive_bar(len(os.listdir())) as bar:
            for image in os.listdir():
                id = image.replace('clean_', '')
                id = id.replace('.jpg','')
                if id in self.image_list:
                    with Image.open(image) as img:
                        img = ToTensor()(img)
                        img = torch.flatten(img)
                        img = img.numpy()
                        self.image_array.append(img)
                        self.image_id.append(id)
                        bar()
            
    def merge_array(self):
        self.image_info['image_id'] = self.image_id
        self.image_info['image_array'] = self.image_array
        self.full_images_info = pd.merge_ordered(self.images,
                                            self.image_info, 
                                            how = 'left',
                                            on ='image_id')
        print(self.full_images_info)
        
        
        
    def run_processor(self):
        self.create_save_folder()
        self.compile_images()
        #self.resize_images()
        self.images_to_array()
        self.merge_array()
        print(self.full_images_info.info())
        os.chdir(f'{self.datapath}/data')
        self.full_images_info.to_pickle(r'clean_fb_image_masters.pkl')
        self.full_images_info.to_pickle(r'clean_image_array.pkl')
        print('\n Data has been saved successfully!')

      

if __name__ == "__main__":
    os.chdir('/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/data')
    data_path = os.getcwd()
    products = pd.read_csv('clean_fb_marketplace_master.csv', lineterminator="\n")
    processor = CreateImageProcessor(products)
    processor.run_processor()

                                                                                                                                                                                         # %%
