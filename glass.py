##importing the data
import pandas as pd
import numpy as np
data= pd.read_csv('E:\\assignment\\knn\\glass.csv')

##checking differnt informations
data.info()

## normalising the data
data_new=(data-data.min())/(data.max()-data.min())

data_new.info()

data_new['Type']=data_new['Type'].astype(np.int64)

##selecting the predictor feature and the taget variable
x=data_new.iloc[:,:9]

y= data_new.iloc[:,9]

##training and testing the dataset
from sklearn.model_selection import train_test_split

##spliting the data into 70%training and 30% testing
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)

##training and testing the dataset
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import accuracy_score

for k in range(25):
    k_value= k+1
    neighbor=knc(n_neighbors=k_value)
    neighbor.fit(x_train,y_train)
    y_pred=neighbor.predict(x_test)
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",k_value)
    
## so we get highest accuracy of  95.38 % for k_value =2    