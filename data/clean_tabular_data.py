#%% 
#Cleaning FB Data
import pandas as pd
import numpy as np
import os


class CleanTabular():
    def __init__(self, products):
        self.products = products
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
        print('Dropping unnecessary columns')
        self.products.drop('Unnamed: 0', inplace=True, axis=1)
        self.products.drop('category', inplace=True,  axis=1)
        self.products.drop('location', inplace=True,  axis=1)
        
    
    """
    Removes the missing and null datafrom the dataframe. 
    """
    def remove_null(self):
        remove_na = self.products.isnull()
        empty_fields = []

        unclean_data = self.products.copy()
        print("Removing Missing Values")
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
        print('Formatting price...')
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
        print("Removing duplicates.")
        columns = ["product_name", "product_description", "location"]
        self.products.drop_duplicates(subset=columns, keep="first")
        


    ''' This function splits the categories into their own columns for future analysis'''
    def split_categories(self):
        print('Splitting the categorical info ...')
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
        print('Splitting city and region information ...')
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
         
        
    ''' This function was designed to run the cleaner and run all the functions embodied above
     and ensures they run in the right order.'''    
    def run_cleaner(self):
        self.drop_duplicates()
        self.split_location() 
        self.split_categories()
        self.format_prices()
        self.remove_null()
        self.drop_columns()
        print("Data was succesfully cleaned. Final output: \n")
        print(self.products.info())
        print(self.products.head())

if __name__ == "__main__":
    data_path = os.getcwd()
    
    # read data from file
    print("Reading Data from Directory")
    products = pd.read_csv(f"{data_path}/Products.csv", lineterminator="\n")
    

    # perform cleaning
    print("Let's clean our data")
    Cleaner = CleanTabular(products)
    Cleaner.run_cleaner()

    # send the data to file
    print("Saving cleaned data to Directory ")
    #os.chdir('data/')
    Cleaner.products.to_csv(r'clean_fb_marketplace_products.csv')
    Cleaner.products.to_json(r'clean_fb_marketplace_products.json')
    print("Successfully cleaned and saved data")
  



# %%


# %%
