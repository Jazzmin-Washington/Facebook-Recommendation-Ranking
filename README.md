# Facebook-Recommendation-Ranking
To develop  a Facebook Marketplace Search Ranking, in which the user develops and trains a multimodal model that accepts images and text. The output of the model generates embedding vectors that are helpful to make recommendations to buyers using their demographic information.  Once the model is trained, it can be deployed, so it’s prepared to accept requests from potential buyers. The way you have to deploy it is by creating an API using FastAPI, containerising it using Docker, and uploading it to an EC2 instance. The API will contain all the models you create and train, so any new request can be processed to make predictions on the cloud. To ensure that the model can be easily modified without building a new image again, the files corresponding to each model are bound to the Docker image, so it allows you to update the models with retrained models even after deploying them.
_______________________________________________________________________________________________________________________
### Project Set-Up

####First step in building the ML models is to retrieve the raw data from AWS Cloud Service. The raw data contains various characters that need to be cleaned and formatted to be used properly. For this project, we were given three files:
       - `Products.csv` which contains the raw product data including: 
                    product description, 
                    price, 
                    product name, 
                    categories,
                    product id
       - `Images.csv` which contains a file containing the 'product_id' and the corresponding 'image_id' relating to  name of the saved images
       - A zip file of the raw jpeg images saved by image_id
_______________________________________________________________________________________________________________________
### Preparing Product Data for Analysis ('clean_tabular_data.py')

#### To make sure the data is usable, several methods were used to clean the data:
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' 
 - Checked and remove any duplicates in the data.
      
             def drop_duplicates(self):
                  print("Removing duplicates... \n")
                  columns = ["image_id"]
                  self.images_products.drop_duplicates(subset=columns, keep="first")        
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' 
 - Format the`price`column by removing the £ symbol and ensuring the price is a numerical float, As well as, removing any outliers that may skew the data
       
              def format_prices(self, max_price: int = 10000, min_price: int = 0.1):
                print('Formatting price...\n')
                self.images_products['price'] = self.images_products['price'].str.replace(r'\£', '', regex = True)
                self.images_products['price'] = self.images_products['price'].str.replace(",", "", regex = True)
                self.images_products['price'] =  pd.to_numeric(self.images_products['price'], downcast="float")

                # Remove outliers
                self.images_products = self.images_products[self.images_products["price"] > min_price]
                self.images_products = self.images_products[self.images_products["price"] < max_price]
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''       
  - The location was split into `Region` and `County` for clear visualisatio. Then the `location` column was then removed. 
       
              def split_location(self):
                print('Splitting city and region information ...\n')   
                self.images_products['County'] = self.images_products['location'].str.split(',').str[1]
                self.images_products['Region'] = self.images_products['location'].str.split(',').str[0]
                self.images_products.drop('location', inplace=True,  axis=1)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''                     
 - Characters that were not ASCII characters were then removed because the models would not be able to interpret these (i.e. emojis)
      
            def clean_text_descriptions(self):
              print('Cleaning Description ...\n')
              self.images_products['product_description'].str.encode('ascii', 'ignore').str.decode('ascii')
              self.images_products['product_name'].str.encode('ascii', 'ignore').str.decode('ascii')
              self.images_products['product_name'] = self.images_products['product_name'].str.split(r'\|').str[0]
 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''          
 - The product categories listed in `category` column was then separated and converted into categorical values using pandas `cat_codes` for eas-er analysis in future. 
      
            def classify_categories(self):
              print('Splitting the categorical info ...\n')
              self.images_products['subcategory_1']= self.images_products['category'].str.split('/').str[1]   
              self.images_products['main_category'] = self.images_products['category'].str.split('/').str[0]
              self.images_products['cat_codes'] = self.images_products['main_category'].astype('category').cat.codes
              self.images_products.drop('category', inplace = True, axis = 1)
 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''                  
- Next, any rows of data that contained null or none values were removed to ensure the data used was fully populated. 
      
            def remove_null(self):
              unclean_data = self.images_products.copy()
              print("Removing Missing Values \n")
              null_options = ['n/a', "N/A", "null", 'None','none', 'NaN', '']
              for n in null_options:
                 if n in unclean_data:
                     unclean_data.replace(n, np.nan)

              unclean_data.dropna(inplace = True)
              self.images_products = unclean_data
  '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
- Finally, the `product` data was then merged with the corresponding `images` directory. 
       
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
                  print(self.images_products.info())
                  
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

###### Upon completion of the  `clean_tabular_script.py` script,  the resulting data were then saved as `clean_fb_marketplace_master.pkl` and `clean_fb_marketplace_master.csv`
--------------------------------------------------------------------------------------------------------------------------
### Preparing Image Data for Analysis ('clean_images.py')

#### The image dataset containes multiple images of the products. It is important that all of these images are the same size for the ML models. `clean_images.py` was created to clean the image dataset.
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
- First, a new folder, `clean_images' was created to save the newly formatted images. 

- `clean_image` function was created to resize which was then copied onto a blank background using `PIL - Pillow`
            
            def clean_image(self, img_name : str):
              # open image
              with Image.open(img_name) as img:

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
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
- `clean_all_images` was then created to iterate through each file in the `images` folder. An `alive-bar` was then added for clear visualisation of progress. 

         def clean_all_images(self):
            self.clean_image_save = f'{self.datapath}/clean_images'
            os.chdir(f'{self.datapath}/images')
            with alive_bar(len(os.listdir())) as bar:
                for image in os.listdir():
                    check = list(os.listdir(f'{self.clean_image_save}'))
                    if '.jpg' in image and image not in check:
                        img  = self. clean_image(image)
                        img.save(f'{self.clean_image_save}/clean_{image}')
                        bar()
            return None
 ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 - Finally, `run_image_cleaner` was created to run all the functions required for the image cleaner. 
 
_______________________________________________________________________________________________________________________
### Create Image Processor to Convert Images to Array (`image_preprocessing.py`)

#### The next step after cleaning the images is to perform the process of converting the images to arrays so ML models can understand the image data. 

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
- First the the 'clean_fb_marketplace_master.csv' file was used to create a DataFrame containing ['image_id', 'product_ids', and 'main_category] information . As well as, compile the 'image_id' into an `image_list` to ensure each of the ids will have a corresponding image array. 

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
- Next, the images were opened using `PIL-Pillow` and  converted into an array using `PILToTensor` from the `PyTorch` module. This array was then flattened and converted into a numpy array using `.numpy()` function. An `alive-bar` was added for visualisation purposes.

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
                              img = PILToTensor()(img)
                              img = torch.flatten(img)
                              img = img.numpy()
                              self.image_array.append(img)
                              self.image_id.append(id)
                              bar()
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
- Finally, the 'image_array' was merged into the DataFrame created during the first step and saved as `clean_fb_image_masters.pkl` and `clean_image_array.pkl with the columns:
    - image_id
    - product_id
    - main_category
    - image_array
_______________________________________________________________________________________________________________________
### Perform Elementary Analysis of Data (Linear Regression and Logistic Regression)




