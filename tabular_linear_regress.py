#%%
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn import metrics


class LinearRegressionMethod():
    def __init__(self):
        os.chdir('/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/data')
        self.products = pd.read_csv(f"clean_fb_marketplace_no_split.csv", lineterminator="\n")
        

    def load_data(self):
        self.x_data = self.products[["product_name", "product_description", "location"]]
        self.y_data = self.products['price']
        print(self.x_data.info())
        print(self.y_data.info())
    def run_data(self):
        X = self.x_data
        
        y = self.y_data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        column_trans =  make_column_transformer(
            (TfidfVectorizer(), 'product_name'),
            (TfidfVectorizer(), 'product_description'),
            (TfidfVectorizer(), 'location'))

        
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

    def run_linear_regress(self):
        self.load_data()
        self.run_data()
        


if __name__ == "__main__":
    LinearRegress = LinearRegressionMethod()
    LinearRegress.run_linear_regress()
    

