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
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' ''''''
 - Checked and remove any duplicates in the data.
      
             def drop_duplicates(self):
                  print("Removing duplicates... \n")
                  columns = ["image_id"]
                  self.images_products.drop_duplicates(subset=columns, keep="first")        
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' 
 - Format the`price`column by removing the £ symbol and ensuring the price is a numerical float, As well as, removing any outliers that may skew the data
       
              def format_prices(self, max_price: int = 10000, min_price: int = 0.1):
                print('Formatting price...\n')
                self.images_products['price'] = self.images_products['price'].str.replace(r'\£', '', regex = True)
                self.images_products['price'] = self.images_products['price'].str.replace(",", "", regex = True)
                self.images_products['price'] =  pd.to_numeric(self.images_products['price'], downcast="float")

                # Remove outliers
                self.images_products = self.images_products[self.images_products["price"] > min_price]
                self.images_products = self.images_products[self.images_products["price"] < max_price]
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' ''''''''''      
  - The location was split into `Region` and `County` for clear visualisatio. Then the `location` column was then removed. 
       
              def split_location(self):
                print('Splitting city and region information ...\n')   
                self.images_products['County'] = self.images_products['location'].str.split(',').str[1]
                self.images_products['Region'] = self.images_products['location'].str.split(',').str[0]
                self.images_products.drop('location', inplace=True,  axis=1)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' ''''''''''                    
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
 
#### To become comfortable with PyTorch and learn basic Machine Learning Models, a linear regression and logistic regression to analyse our data. 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 1. Linear Regression - Using `['product_name', 'product_description', 'location']` as the X-value and `price` as the Y value,
 
         def load_data(self):
               self.x_data = self.products[["product_name", "product_description", "location"]]
               self.y_data = self.products['price']
               print(self.x_data.info())
              print(self.y_data.info())
              
 1a.  Using `scikit-learn` module, X and Y were split into test and train sets. 
 
           def run_data(self):
               X = self.x_data

               y = self.y_data

               X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
               
1b. A `TfidVectorizer` was used to analyse the text in our X-values to convert them into values. 
                      
                      column_trans =  make_column_transformer(
                          (TfidfVectorizer(), 'product_name'),
                          (TfidfVectorizer(), 'product_description'),
                          (TfidfVectorizer(), 'location'))

1c. Linear Regression were then run using `scikit-learn` or `sklearn` by creating a pipeline followed by fitting the pipeline and then running a prediction.

               # define a pipeline
               pipe = make_pipeline(column_trans, LinearRegression())
               print(f'Making Pipeline : {pipe}')

              # Fit pipeline
               fit = pipe.fit(X_train, y_train)
               print(f"Fitting pipeline: {fit}")

               # Make prediction based on pipeline:
               y_pred = pipe.predict(X_test)
               print(f'Making predictions: {y_pred}')

               # Print Predictions
               print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
               print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
               print('Root Mean Squared Error:', metrics.r2_score(y_test, y_pred))

1d. Results:


![image](https://user-images.githubusercontent.com/102431019/202288371-01959b20-443b-4efd-b9c3-fcbc7d1ba96f.png)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

2. Logistic Regression - Image Classifier - Using `image_array` column as features and `cat_codes` as label values, a logistic regression was run to create classification report using `sklearn` module. 

2a. First, the `main_category` column was used to create a `cat_codes` column by using sklearn's `Label Encoder`. This file was then saved for future use as `image_decoder.pkl`. 

       le = LabelEncoder
       class ClassifierLogRegress():
           def __init__(self, data):
               self.groupXY = []
               self.image_array  = data 
               pkl_file = open('image_decoder.pkl', 'rb')
               self.le_category = pickle.load(pkl_file)
               pkl_file.close()
               self.image_array['cat_codes'] = self.le_category.transform(self.image_array['main_category'].astype('category'))   
               self.image_array = self.image_array.sort_values('cat_codes', ascending = True)
               print(self.image_array.head(4))

2b. `image_array` was then set to features and the corresponding `cat_codes` was then set to labels corresponding to X, Y values, respectively. Features and labels were then tupled together and appended to a list.

       for i in range(len(self.image_array)):
                   features = self.image_array['image_array'][i]
                   labels = self.image_array['cat_codes'][i]
                   grouped = (features, labels)
                   create_tuple = tuple(grouped)
                   self.groupXY.append(create_tuple)
                   
2c. The resulting X,Y array needed to be formatted for the image size using `np.zeros` 
       
       n = len(self.groupXY)
        image_size = 100
        array_size = int((image_size **2)*3)
        self.X = np.zeros((n, array_size))
        self.y = np.zeros(n)

        for arrays in range(n):
            features, labels = self.groupXY[arrays]
            self.X[arrays, :] = features
            self.y[arrays] = labels
            
  2d. Using `sklearn` module, X and Y were then split into train and test sets before being fit and used to predict labels given the features. This makes a very rudimentary image classifer. 
  
         def run_logreg_class(self): 
               model = LogisticRegression(max_iter=100)
               X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3,random_state=0)
               model.fit(X_train, y_train)
               pred = model.predict(X_test)
               print ('accuracy:', accuracy_score(y_test, pred))
               print('report:', classification_report(y_test , pred))
 
 2e. Results: 
 
 
![image](https://user-images.githubusercontent.com/102431019/202295186-cb35d1ea-70a4-49c8-be42-e3f242e2b151.png)

_______________________________________________________________________________________________________________________
### Create Image Dataset (`Image_Dataset.py`)

#### An Image Dataset was created in order to build a dataloaders to load data for future Machine Learning Models.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
1. A decoder and encoder needs to be created or loaded using saved `Label Encoder` values to allow easy conversion of category values to label values and vice versa. 

2. Images were then processed and transformed using `transforms.compose` to undergo changes such as resizing, center crop, normalization and converting the resulting arrays into tensors

3.`__getitem__` method was used to retrieve label values and feature (transformed image arrays) values for each set of data: 

               def __getitem__(self, index:int):
                      label = self.labels[index]
                      label = self.encode[label]
                      label = torch.as_tensor(label)
                      os.chdir(f'{self.datapath}/clean_images')
                      image = Image.open(f'clean_{self.image_id[index]}.jpg')
                      if image.mode != 'RGB':
                          feature = self.transformer_Gray(image)
                      else:
                          feature = self.transformer(image)

                      return feature, label
                      
 4. `__len__` method was used to retrieve the length of the dataset
 
         def __len__(self):
               return(len(self.data.image_id))
               
 5. The Image_Dataset was then tested to ensure data batches could be retrieved. 
 
        image_loader = DataLoader(new_fb, batch_size = 32, shuffle = True, num_workers = 1)
           for i, (features,labels) in enumerate(image_loader):
               print(features)
               print(labels)
               print(features.size())
               if i == 0:
                   break
_______________________________________________________________________________________________________
### Building a Convolutional Neural Network (`CNN_model_base.py`)

#### To understand the dynamics of neural networks, a convolution network was built to classify images. 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
1. `torch.nn.Module` was used to create a CNNBuild class consisting of the architecture for the CNN network. A convolutional network consist of layers of `conv2d` followed by `torch.nn.linear` layers. As well as, `torch.nn.ReLU` layers inbetween each layer. A forward function is then created to process each batch of features through the model.  Finally, a probability is then produced. 

       class CNNBuild(torch.nn.Module):
           def __init__(self):
               super().__init__()
               self.cnn_layers = torch.nn.Sequential(
                   torch.nn.Conv2d(3, 8, 7),
                   torch.nn.ReLU(),
                   torch.nn.Conv2d(8, 16, 7),
                   torch.nn.ReLU(),
                   torch.nn.Flatten(),
                   torch.nn.Linear(215296, 1200),
                   torch.nn.ReLU(),
                   torch.nn.Linear(1200, 600),
                   torch.nn.ReLU(),
                   torch.nn.Linear(600, 300),
                   torch.nn.ReLU(),
                   torch.nn.Linear(300, 128),
                   torch.nn.ReLU(),
                   torch.nn.Linear(128,13),
                   )
           def forward(self, features):
              return self.cnn_layers(features)

           def predict_probs(self, features):
               with torch.no_grad():
                   return self.forward(features)

2. `def train(model, epoch = 10)` is created to train the model. A summary writer, optimiser and criterion is passed into the model to allow for tuning purposes. Both accuracy and loss values are kept and graphed using `TensforFlow`. 

       def train(model, epoch = 10):
           #device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
           #model.to(device)
           writer = SummaryWriter()
           optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
           criterion = torch.nn.CrossEntropyLoss()
           batch_idx = 0
           for epochs in range(epoch):
               for batch in train_loader:
                   features, labels = batch
                   #features = features.to(device)
                   #labels = labels.to(device)
                   prediction = model(features)
                   loss = criterion(prediction, labels)
                   loss.backward()
                   print(loss.item())
                   train_accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
                   optimiser.step()
                   optimiser.zero_grad()
                   writer.add_scalar("Train Loss", loss.item(), batch_idx)
                   writer.add_scalar("Train Accuracy", train_accuracy, batch_idx)
                   batch_idx += 1

               model_save_name = f'{path}/image_model_evaluation/image_cnn.pt'
               torch.save(model.state_dict(), model_save_name)

3. Finally, the dataloader is created to iterate through each of the batches. 

        if __name__ == '__main__':
           dataset = FbMarketImageDataset()
           train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers = 1)
           model = CNNBuild()
           train(model)
4. Results: 
