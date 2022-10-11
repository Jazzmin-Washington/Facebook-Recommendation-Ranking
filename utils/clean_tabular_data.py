lass CleanTabularandImages():
    def __init__(self, products, images):
        self.products = products
        self.images = images
        print('Unclean Data:')
        print( self.products.info())
        self.images.info()
        
    
    """
    Removes unnecessary columns and merges the image data with product data to create a master
    document
    """
    def drop_columns_w_merge(self):
        print('Dropping unnecessary columns \n')
        self.products.drop('Unnamed: 0', inplace=True, axis=1)
        self.images.drop('Unnamed: 0', inplace=True, axis=1)
        self.products.rename(columns = {'id':'product_id'}, inplace = True)
        self.images.rename(columns = {'id':'image_id'}, inplace = True)
        self.images_products = pd.merge_ordered(self.products, 
                                                self.images,
                                                on = 'product_id',
                                                how='left')

    
    """
    Removes the missing and null datafrom the dataframe. 
    """
    def remove_null(self):
        unclean_data = self.images_products.copy()
        print("Removing Missing Values \n")
        null_options = ['n/a', "N/A", "null", 'None','none', 'NaN', '']
        for n in null_options:
           if n in unclean_data:
               unclean_data.replace(n, np.nan)

        unclean_data.dropna(inplace = True)
        self.images_products = unclean_data
        

    """
    Converts price column into a float within the dataframe and removes outliers 
    """
    def format_prices(self, max_price: int = 10000, min_price: int = 0.1):
        print('Formatting price...\n')
        self.images_products['price'] = self.images_products['price'].str.replace(r'\£', '', regex = True)
        self.images_products['price'] = self.images_products['price'].str.replace(",", "", regex = True)
        self.images_products['price'] =  pd.to_numeric(self.images_products['price'], downcast="float")
        
        # Remove outliers

        self.images_products = self.images_products[self.images_products["price"] > min_price]
        self.images_products = self.images_products[self.images_products["price"] < max_price]
        
    """
    Removes the duplicates from the dataframe. 
    """
    def drop_duplicates(self):
        print("Removing duplicates... \n")
        columns = ["image_id"]
        self.images_products.drop_duplicates(subset=columns, keep="first")
        

    """ 
    This function splits the categories into their own columns and groups the main
    category into separate classification numbers
    """

    def classify_categories(self):
        print('Splitting the categorical info ...\n')
        self.images_products['subcategory_1']= self.images_products['category'].str.split('/').str[1]   
        self.images_products['main_category'] = self.images_products['category'].str.split('/').str[0]
        self.images_products['cat_codes'] = self.images_products['main_category'].astype('category').cat.codes
        self.images_products.drop('category', inplace = True, axis = 1)
        
        
    """
    This function splits the city and region into separate columns and allows for the removal
    of entries that do not have a city and regions in the drop_columns function
    """

    def split_location(self):
        print('Splitting city and region information ...\n')   
        self.images_products['County'] = self.images_products['location'].str.split(',').str[1]
        self.images_products['Region'] = self.images_products['location'].str.split(',').str[0]
        self.images_products.drop('location', inplace=True,  axis=1)

    """This function converts all the text in product name and description to ascii characters
    """ 
    def clean_text_descriptions(self):
        print('Cleaning Description ...\n')
        self.images_products['product_description'].str.encode('ascii', 'ignore').str.decode('ascii')
        self.images_products['product_name'].str.encode('ascii', 'ignore').str.decode('ascii')
        self.images_products['product_name'] = self.images_products['product_name'].str.split(r'\|').str[0]
       
        
    """ 
    This function was designed to run the cleaner and run all the functions embodied above
     and ensures they run in the right order.
     """   
     
    def run_cleaner(self):
        self.drop_columns_w_merge()
        self.drop_duplicates()
        self.split_location() 
        self.clean_text_descriptions()
        self.format_prices()
        self.classify_categories()
        self.remove_null()
        print("Data was succesfully cleaned. Final output: \n")
        print(self.images_products.info(), self.images_products.head(5))
       


if __name__ == "__main__":
    data_path = '/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/data'
    os.chdir(data_path)

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
    os.chdir(data_path)
    Cleaner.images_products.to_csv(r'clean_fb_marketplace_master.csv')
    Cleaner.images_products.to_json(r'clean_fb_marketplace_master.json')
    Cleaner.images_products.to_pickle(r'clean_fb_marketplace_images_master.pkl')
    print("Successfully cleaned and saved data")


