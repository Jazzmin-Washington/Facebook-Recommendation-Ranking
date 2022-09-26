#%%
# Create and Image Classifier
import numpy as np
import os
import pandas as pd
import torch
import sklearn
from numpy import asarray
from PIL import Image
from torchvision.transforms import ToTensor
from torch import flatten
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from alive_progress import alive_bar
from tqdm import tqdm



class CreateImageClassifier():
    def __init__(self, products, final_image_size: int = 512):
        self.products = products
        self.image_array = []
        self.datapath = '/home/jazz/Documents/AiCore_Projects/Facebook-Marketplace-Ranking'
        self.final_image_size = final_image_size
        self.image_list = pd.DataFrame()
        self.image_array_full = []
        
    def create_save_folder(self):
        if not os.path.exists(f'{self.datapath}/clean_images'):
            os.makedirs (f'{self.datapath}/clean_images' , exist_ok=True)
        
    def compile_images(self):
        self.second_image = self.products['image_2']
        self.second_image.dropna(inplace = True)
        self.images= list(self.second_image) + list(self.products['image_1'])
        print(len(self.products['image_1']) + len(self.second_image))
        self.image_list['image_id'] = self.images
       

    def clean_image(self, img_name : str):
            # open image
            img = Image.open(img_name)

            # resize by finding the biggest side of the image and calculating ratio to resize by
            max_side_length = max(img.size)
            resize_ratio = self.final_image_size / max_side_length
            img_width = int(img.size[0]*resize_ratio)
            img_height = int(img.size[1]*resize_ratio)
            img = img.resize((img_width, img_height))

            # convert to rgb
            img = img.convert("RGB")

            # paste on black image
            final_img = Image.new(mode="RGB", size=(
                self.final_image_size, self.final_image_size))
            final_img.paste(img, ((self.final_image_size - img_width) //
                            2, (self.final_image_size - img_height)//2))

            return final_img

    def process_all_images(self):
        self.clean_image_save = f'{self.datapath}/clean_images'
        os.chdir(f'{self.datapath}/images')
        with alive_bar(len(os.listdir())) as bar:
            for image in os.listdir():
                file_name = str(image).replace('clean_', "")
                id = file_name.replace('.jpg', '')
                if '.jpg' in image and image not in self.clean_image_save and id in self.images:
                    img  = self.clean_image(image)
                    img.save(f'{self.clean_image_save}/clean_{image}')
                    img_to_tensor = self.image_to_tensor(image)
                    img_info = {'image_id': id,
                                'array': img_to_tensor}
                    self.image_array.append(img_info)
                    bar()
        self.images_w_array = pd.DataFrame(self.image_array)
        self.images_w_array.to_csv(r'clean_image_array.csv')

        return None
    
    def image_to_tensor(self, img_name):   
        image = Image.open(img_name)
        t = ToTensor()
        tensor_image = t(image)
        flattened = torch.flatten(tensor_image) 
        flat_numpy = flattened.numpy()
        
    
        return flat_numpy
    
    def run_classifiers(self):
        self.create_save_folder()
        self.compile_images()
        self.process_all_images()
        
    

if __name__ == "__main__":
    os.chdir('/home/jazz/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/data')
    data_path = os.getcwd()
    products = pd.read_csv('clean_fb_marketplace_products.csv', lineterminator="\n")
    classifier = CreateImageClassifier(products)
    classifier.run_classifiers()
# %%

# %%
