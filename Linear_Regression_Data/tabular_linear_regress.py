#%%
# Import Dependencies
from re import X
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class LinearRegressionMethod():
    def __init__(self):
       self.products = ''
    
    def load_data(self):
        os.chdir('/home/jazz/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/data')
   
        # Read Data from file
        print("Reading Data from Directory")
        self.products = pd.read_csv(f"clean_fb_marketplace_products.csv", lineterminator="\n")
        
        self.products.drop(['Unnamed: 0', 
                            'id',
                            'category_1',
                            'category_2',
                            'category_3',
                            'category_4',
                            'category_5'],
                             inplace=True,  axis=1)
        self.products.info()
        print(self.products)
        self.y = self.products['price']
        print(self.y)
        remove_price = self.products.drop('price', axis =1)
        self.X = remove_price


    def prep_data(self):
        vectorizer = TfidfVectorizer()
        x_vector = vectorizer.fit_transform(self.X)
        y_vector = vectorizer.fit_transform(self.y)
       
        X = x_vector
        y = y_vector
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        linear_regression_model = LinearRegression().fit(X_train_std, y_train)
        y_prediction = linear_regression_model.predict(X_test_std)
        r2_score(y_test, y_prediction)



    def run_linear_regress(self):
        self.load_data()
        self.prep_data()
        #self.data_extraction()


if __name__ == "__main__":
    LinearRegress = LinearRegressionMethod()
    LinearRegress.run_linear_regress()
