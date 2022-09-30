#%%
# Import Dependencies
from re import X
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class LinearRegressionMethod():
    def __init__(self):
       self.products = ''
    
    def run_linear_regress(self):
        os.chdir('/home/jazz/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/data')
   
        # Read Data from file
        print("Reading Data from Directory")
        self.products = pd.read_csv(f"clean_fb_marketplace_products.csv", lineterminator="\n")
        self.data = self.products[['price']]
        
        # Define matrix x and response vector y:
        y = self.products.price
        remove_price = self.products.drop('price', axis =1)
        X = remove_price

        #Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
        # Make column transform to transform text columns of feature matrix
        column_trans = make_column_transformer(
        (TfidfVectorizer(), "product_name"),
        (TfidfVectorizer(), "product_description"),
        (TfidfVectorizer(), "location"))
        
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
        print(f'RMSE of model = {metrics.mean_squared_error(y_test, y_pred, squared = False)}')
        print(f' R2 Score = {metrics.r2_score(y_test, y_pred)}')

if __name__ == "__main__":
    LinearRegress = LinearRegressionMethod()
    LinearRegress.run_linear_regress()
