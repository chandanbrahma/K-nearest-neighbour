## importing the data
import pandas as pd
data= pd.read_csv('E:\\assignment\\knn\\Zoo.csv')
data.info()



## segrigating the target into prectors and target variables
x= data.iloc[:,1:17]

y= data.iloc[:,-1]


##dividng  the train and test dataset into 70% and 30% respectively
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3)


##training and testing the model using KNN algorithem
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import accuracy_score


for k in range (30):
    k_value=k+1
    neighbor= knc(n_neighbors=k_value)
    neighbor.fit(x_train,y_train)
    y_pred= neighbor.predict(x_test)
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",k_value)

##so for k=4 we are getting accraacy of 90.3%