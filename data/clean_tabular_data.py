#%% 
#Cleaning FB Data
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


class CleanTabularandImages():
    def __init__(self, products, images):
        self.products = products
        self.images = images
        print('Unclean Data:')
        print( self.products.info())
        self.image_list_full =[]
       
    
    """
    Removes unnecessary columns 
    """
    def drop_columns(self):
        print('Dropping unnecessary columns \n')
        self.products.drop('Unnamed: 0', inplace=True, axis=1)
        self.images.drop('Unnamed: 0', inplace=True, axis=1)
        self.products.drop('category', inplace = True, axis = 1)
        self.products.drop('location', inplace=True,  axis=1)
        
    
    """
    Removes the missing and null datafrom the dataframe. 
    """
    def remove_null(self):
        remove_na = self.products.isnull()
        unclean_data = self.products.copy()
        print("Removing Missing Values \n")
        null_options = ['n/a', "N/A", "null", 'None','none', 'NaN', '']
        for n in null_options:
           if n in unclean_data:
               unclean_data.replace(n, np.nan)

        unclean_data.dropna(inplace = True)
        self.products = unclean_data
        

    """
    Converts price column into a float within the dataframe
    """
    def format_prices(self, max_price: int = 10000, min_price: int = 0.1):
        print('Formatting price...\n')
        self.products['price'] = self.products['price'].str.replace(r'\£', '', regex = True)
        self.products['price'] = self.products['price'].str.replace(",", "", regex = True)
        self.products['price'] =  pd.to_numeric(self.products['price'], downcast="float")
        
        # Remove outliers

        self.products = self.products[self.products["price"] > min_price]
        self.products = self.products[self.products["price"] < max_price]
        
    """
    Removes the duplicates from the dataframe. 
    """
    def drop_duplicates(self):
        print("Removing duplicates... \n")
        columns = ["product_name", "product_description", "location"]
        self.products.drop_duplicates(subset=columns, keep="first")
        

    ''' This function splits the categories into their own columns for future analysis'''
    def split_categories(self):
        print('Splitting the categorical info ...\n')
        self.products['subcategory_1']= self.products['category'].str.split('/').str[1]   
        self.products['main_category'] = self.products['category'].str.split('/').str[0]

    ''' This function splits the city and region into separate columns and allows for the removal
        of entries that do not have a city and regions in the drop_columns function'''
    def split_location(self):
        print('Splitting city and region information ...\n')   
        self.products['County'] = self.products['location'].str.split(',').str[1]
        self.products['Region'] = self.products['location'].str.split(',').str[0]


    def clean_text_descriptions(self):
        print('Cleaning Description ...\n')
        self.products['product_description'].str.encode('ascii', 'ignore').str.decode('ascii')
        self.products['product_name'].str.encode('ascii', 'ignore').str.decode('ascii')
        self.products['product_name'] = self.products['product_name'].str.split(r'\|').str[0]
       
       
    
    '''This function matches the 'id' column within the Products.csv to the 
    'product_id' column within the Images.csv so the corresponding pictures can be matched to
    the products''' 
    def match_pic_id(self):
        print('Matching product id with image identification...\n')
        self.image_list = []
        id_list = list(self.products['id'])
        for i in tqdm(range(len(id_list)), "Matching Product ID to Images:"):
            id = str(id_list[i])
            product_list = list(self.images['product_id'])
            if id in product_list:
                self.image_list.append(id)
            else:
                self.image_list.append(np.nan)


    ''' This function will add the image column for the list of images related to the product. '''
    def create_image_column(self):
        image_column =[]
        id_list = list(self.products['id'])
        for i in tqdm(range(len(id_list)), "Making Image Columns for Products:"):
            id = str(id_list[i])
            image_match = ''
            matching_id = self.images['product_id'].str.contains(id)
            self.image_match = self.images['id'][matching_id] 
            image_match = list(self.image_match)
            try:            
                image_column.append(image_match[0])
            except:
                image_column.append(np.nan)
            try:
                self.image_list_full.append(image_match[1])
            except:
                self.image_list_full.append(np.nan)


        self.products['image_1'] = image_column
        self.products['image_2'] = self.image_list_full
        self.products.dropna(subset = ['image_1'], inplace = True)
        
        
        

        
    ''' This function was designed to run the cleaner and run all the functions embodied above
     and ensures they run in the right order.'''    
    def run_cleaner(self):
        self.drop_duplicates()
        self.split_location() 
        self.clean_text_descriptions()
        self.format_prices()
        self.match_pic_id()
        self.split_categories()
        self.remove_null()
        self.create_image_column()
        self.drop_columns()
        print("Data was succesfully cleaned. Final output: \n")
        print(self.products.info())
        print(self.products.head())
        



if __name__ == "__main__":
    os.chdir('/home/jazz/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/data')
    data_path = os.getcwd()
    
    # read data from file

    print("Reading Data from Directory")
    products = pd.read_csv(f"{data_path}/Products.csv", lineterminator="\n")
    images = pd.read_csv(f"{data_path}/Images.csv")

    # perform cleaning
    print("Let's clean our data")
    Cleaner = CleanTabularandImages(products , images)
    Cleaner.run_cleaner()

    # send the data to file
    print("Saving cleaned data to Directory ")
    #os.chdir('data/')
    Cleaner.products.to_csv(r'clean_fb_marketplace_products.csv')
    Cleaner.products.to_csv(r'clean_image_data.csv')
    Cleaner.products.to_json(r'clean_fb_marketplace_products.json')
    Cleaner.products.to_pickle(r'clean_fb_marketplace_products.pkl')
    print("Successfully cleaned and saved data")
  


