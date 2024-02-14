import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.neural_network import MLPClassifier
#server imports
import socket
import os

data = pd.read_csv('dataset.csv',sep=';').fillna(0)
data


# ### Data pre-processing & Statistical Analysis

# In[3]:


data=data.iloc[:,1:]
data.describe()


# In[4]:


X=data.iloc[:,1:7]
Y=data.iloc[:,0]


# In[5]:


X.describe()


# In[6]:


Y.describe()


# ### Splitting data into train, test, and predict dataset

# In[7]:


X,data_val_X,Y,data_val_Y = train_test_split(X, Y, train_size=0.8,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7,random_state=1)


# ### Training MLP classifier with three layers MLP classifier with 6neutron x 5neutron x 3neutron
# 
# The model acheived an accuracy of upto 99% and the confusion matrix also shows the number of True positive = 10935 and True negative = 495, against False positive =12 and False negative = 80

# In[8]:

def trainML():
    clf = MLPClassifier(activation='relu', hidden_layer_sizes=(6,5,3),max_iter=500)
    clf.fit(X_train, y_train)
    print("Training Score:", clf.score(X_train,y_train))
    print("Test Score:", clf.score(X_test,y_test))
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    print("CV Score:", np.mean(scores))
    metrics.ConfusionMatrixDisplay.from_estimator(clf,X_test,y_test)
    plt.show()
    svc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
    return clf

def pridiction(clf,data_val_X):

    y_pred=clf.predict(data_val_X)
    print("Validation Score:",clf.score(data_val_X,data_val_Y) )
    scores = cross_val_score(clf, data_val_X,data_val_Y, cv=5)
    print("Cross validation Score:", np.mean(scores))
    metrics.ConfusionMatrixDisplay.from_estimator(clf,data_val_X,data_val_Y)
    plt.show()
    svc_disp = RocCurveDisplay.from_estimator(clf, data_val_X, data_val_Y)

    print(classification_report(data_val_Y, y_pred))

    return y_pred


def start_server():
    host = '127.0.0.1'
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    #trian the ML
    print("Training the ML Server: ")
    clf = trainML()

    #inform user Server listning
    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")

        data = client_socket.recv(1024)

        if not data:
            break

        received_value = data.decode('utf-8')
        print(f"Received value: {received_value}")

        data_val_X = received_value
        #data_val_Y = 0.8
        y_pridction = pridiction(clf,data_val_X)
        print(f"final y_pridction:  {y_pridction}")

        #close Server socket
        client_socket.close()

if __name__ == "__main__":
    start_server()