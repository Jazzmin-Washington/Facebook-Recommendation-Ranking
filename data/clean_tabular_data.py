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
        self.city = []
        self.region = []
        self.cat_list_1 = []
        self.cat_list_2 = []
        self.cat_list_3 = []
        self.cat_list_4 = []
        self.cat_list_5 = []
    
    """
    Removes the index column titled: Unnamed: 0 from the dataframe. 
    """
    def drop_columns(self):
        print('Dropping unnecessary columns \n')
        self.products.drop('Unnamed: 0', inplace=True, axis=1)
        self.products.drop('category', inplace=True,  axis=1)
        self.images.drop('Unnamed: 0', inplace=True, axis=1)
        #self.products.drop('location', inplace=True,  axis=1)
        
    
    """
    Removes the missing and null datafrom the dataframe. 
    """
    def remove_null(self):
        remove_na = self.products.isnull()
        empty_fields = []

        unclean_data = self.products.copy()
        print("Removing Missing Values \n")
        null_options = ['n/a', "N/A", "null", 'None','none']
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
        for i in range(self.len_products):
            cat_list = ""
            cat_list = self.products['category'][i].split('/')
            try:    
                self.cat_list_1.append(cat_list[0])
                self.cat_list_2.append(cat_list[1])
                self.cat_list_3.append(cat_list[2])
            except IndexError:
                self.cat_list_3.append('/')
            try:
                self.cat_list_4.append(cat_list[3])
            except IndexError:
                self.cat_list_4.append('/')
            try:
                self.cat_list_5.append(cat_list[4])
            except IndexError:
                self.cat_list_5.append('/')

        self.products['category_1'] = self.cat_list_1
        self.products['category_2'] = self.cat_list_2
        self.products['category_3'] = self.cat_list_3
        self.products['category_4'] = self.cat_list_4
        self.products['category_5'] = self. cat_list_5
       

    ''' This function splits the city and region into separate columns and allows for the removal
        of entries that do not have a city and regions in the drop_columns function'''
    def split_location(self):
        print('Splitting city and region information ...\n')
        self.len_products = len(self.products)
        for i in range(self.len_products):
            location = self.products['location'][i].split(',')
            try:
                self.city.append(location[0])
                self.region.append(location[1])
            except:
                self.region.append(np.nan)      
        self.products['City'] = self.city
        self.products['Region'] = self.region
        self.products.drop('location',  axis=1)
         
    def clean_description(self):
        print('Cleaning Description ...\n')
        self.products['product_description'].str.replace(r'/[^a-zA-Z0-9]/g', '', regex = True)
      
    
    '''This function matches the 'id' column within the Products.csv to the 
    'product_id' column within the Images.csv so the corresponding pictures can be matched to
    the products''' 
    
    def match_pic_id(self):
        print('Matching product id with image identification...\n')
        self.image_list = []
        id_list = list(self.products['id'])
        for i in tqdm(range(len(id_list)), "Matching Product ID to Images:"):
            id = str(id_list[i])
            matching_id = self.images['product_id'].str.contains(id)
            image_match = self.images['id'][matching_id]
            image_match = list(image_match)
            image_list = {'image_matches': image_match,
                        'product_id': id}
            self.image_list.append(image_list)
        self.image_matches = pd.DataFrame.from_dict(self.image_list)
        print(self.image_matches.info())

    ''' This function will add the image column for the list of images related to the product. '''
    def merge_image_columns(self):
        self.products['images'] = self.image_matches['image_matches']
        print(self.products.info())
        
    
    ''' This function was designed to run the cleaner and run all the functions embodied above
     and ensures they run in the right order.'''    
    def run_cleaner(self):
        self.drop_duplicates()
        self.split_location() 
        self.split_categories()
        self.clean_description()
        self.format_prices()
        self.remove_null()
        self.drop_columns()
        self.match_pic_id()
        self.merge_image_columns()
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
    Cleaner.products.to_json(r'clean_fb_marketplace_products.json')
    Cleaner.products.to_pickle(r'clean_fb_marketplace_products.pkl')
    print("Successfully cleaned and saved data")
  


