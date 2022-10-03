#%%
#
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class ClassifierLinRegress():
    def __init__(self, data):
        self.groupXY = []
        self.image_array  = data
   
        
    def lin_regress_setup(self):
        print(self.image_array.info())
        for i in range(len(self.image_array)):
            features = self.image_array['array'][i]   
            labels = self.image_array['cat_codes'][i]
            grouped = (features, labels)
            create_tuple = tuple(grouped)
            print(create_tuple)
            self.groupXY.append(create_tuple)
       
       
        n = len(self.groupXY)
        image_size = 100
        array_size = int((image_size **2)*3)
        self.X = np.zeros((n, array_size))
        self.y = np.zeros(n)

        for arrays in range(n):
            features, labels = self.groupXY[arrays]
            self.X[arrays, :] = features
            self.y[arrays] = labels
        
    def run_linreg_class(self): 

        model = LogisticRegression(max_iter=100)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3,random_state=0)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print ('accuracy:', accuracy_score(y_test, pred))
        print('report:', classification_report(y_test, pred))


if __name__ == "__main__":
    data = pd.read_pickle(r'clean_image_array.pkl')
    class_linreg = ClassifierLinRegress(data)
    print('Great Setup is finished \n')
    print(' Running Linear Regression... \n')
    class_linreg.lin_regress_setup()
    class_linreg.run_linreg_class()
   

# %%
