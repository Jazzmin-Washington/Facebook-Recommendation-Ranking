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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import flatten
from torchvision import transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm
from clean_images import CleanImages


class CreateImageProcessor():
    def __init__(self, products):
        self.products = products
        self.image_array = []
        self.full_images_array = pd.DataFrame()
        self.datapath = '/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking'
        self.clean_image_save = f'{self.datapath}/clean_images'
    
        
    def create_save_folder(self):
        if not os.path.exists(f'{self.datapath}/clean_images'):
            os.makedirs (f'{self.datapath}/clean_images' , exist_ok=True)
        
    def compile_images(self):
        self.second_image = self.products['image_2']
        self.second_image.dropna(inplace = True)
        self.images = list(self.products['image_1']) + list(self.second_image)
        print(len(self.images))
        self.products.drop('Unnamed: 0', inplace=True, axis=1)
        self.products.info()
    

    def resize_images(self):
            cleaner = CleanImages()
            print("Let's begin reformatiing the images...")
            cleaner.run_image_cleaner()
            print(f"Success! Reformatted Images were saved at: {cleaner.clean_image_save} ")

    def images_to_array(self):
        print('Converting Images to Numpy Array...')
        os.chdir(self.clean_image_save)
        with alive_bar(len(os.listdir())) as bar:
            for image in os.listdir():
                id = image.replace('clean_', '')
                id = id.replace('.jpg','')
                if id in self.images:
                    with Image.open(image) as img:
                        img = ToTensor()(img)
                        img = torch.flatten(img)
                        img = img.numpy()
                        self.image_array.append(img)
                        bar()

            
    def get_category(self):
        cat_list = []
        cat_codes_list = []
        product_list = []
        self.products['image_array_1'] = self.products['image_1'].copy()
        self.products['image_array_2'] = self.products['image_2'].copy()
        for i in tqdm(range(len(self.images)), "Adding category data:"):
            id = str(self.images[i])
            array = str(self.image_array[i])
            if id in self.second_image:
                    matching_id = self.products['image_2'].str.contains(id)
                    self.products['image_array_2'].str.replace(id,  array)
            elif id in list(self.products['image_1']):
                    matching_id =self.products['image_1'].str.contains(id)
                    self.products['image_array_1'].str.replace(id, array)
            else:
                pass
            product_match = str(self.products['id'][matching_id]).split('    ')[1]
            product_match = product_match.split('\n')[0] 
            cat_match = str(self.products['main_category'][matching_id]).split('    ')[1] 
            cat_match = cat_match.split('\n')[0] 
            cat_codes = str(self.products['cat_codes'][matching_id]).split('    ')[1] 
            cat_codes = cat_codes.split('\n')[0] 
            product_list.append(product_match) 
            cat_list.append(cat_match)
            cat_codes_list.append(cat_codes)
        print(cat_codes_list[:4])
        self.full_images_array['product_id'] = product_list
        self.full_images_array['image_id'] = self.images
        self.full_images_array['array'] = self.image_array
        self.full_images_array['main_category'] = cat_list
        self.full_images_array['cat_codes'] = cat_codes_list
        
        

    def run_classifiers(self):
        self.create_save_folder()
        self.compile_images()
        #self.resize_images()
        self.images_to_array()
        self.get_category()
        print(self.full_images_array.info())
        os.chdir(f'{self.datapath}/data')
        self.products.to_pickle(r'clean_fb__image_info.pkl')
        self.full_images_array.to_pickle(r'clean_image_array.pkl')
        print('\n Data has been saved successfully!')
       

class ClassifierLinRegress():
    def __init__(self, data):
        self.groupXY = []
        self.image_array  = data
        self.features = []
        self.labels = []
        
    def lin_regress_setup(self):
        print(self.image_array.info())
        for i in range(len(self.image_array)):
            features = self.image_array['array'][i]  
            labels = self.image_array['cat_codes'][i]
            self.features.append(features)
            self.labels.append(labels)

            # Split datasets for training/testing
        X = self.features
        y = self.labels
        print(X[:5], y[:5])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        log_reg = LogisticRegression()

        # Train the model
        log_reg.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = log_reg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification = classification_report(y_test, y_pred)
        print ('accuracy:', accuracy)
        print('report:', classification)
      

if __name__ == "__main__":
    os.chdir('/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/data')
    data_path = os.getcwd()
    #products = pd.read_csv('clean_fb_marketplace_products.csv', lineterminator="\n")
    #classifier = CreateImageProcessor(products)
    #classifier.run_classifiers()
    data = pd.read_pickle(r'clean_image_array.pkl')
    #data = classifier.full_images_array
    class_linreg = ClassifierLinRegress(data)
    print('Great Setup is finished \n')
    print(' Running Linear Regression... \n')
    class_linreg.lin_regress_setup()
    class_linreg.run_linreg_class()
    
   

# %%
