# Start

import pandas as pd  
import numpy as np  
#import matplotlib.pyplot as plt  
#import seaborn as sns
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
    
    print("Done Training!")

    return clf

def pridiction(clf,data_val_X):

    y_pred=clf.predict(data_val_X)
    print("Validation Score:",clf.score(data_val_X,y_pred) )
    scores = cross_val_score(clf, data_val_X,y_pred, cv=5)
    print("Cross validation Score:", np.mean(scores))

    return y_pred

def send_array(host, port, array):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.sendto((host, port))
    array_bytes = array.tobytes()
    client_socket.send(array_bytes)
    client_socket.close()



def start_server():
    host = '192.168.2.2'
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    #trian the ML
    print("Training the ML Server: ")
    clf = trainML()

    #inform user Server listning
    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, client_address = server_socket.recvfrom()
        print(f"Connection from {client_address}")

        received_value = client_socket.recv(1024)

        if not received_value:
            break

        #print(f"Received value: {received_value}")
        #print(type(received_value))

        # Convert bytes to NumPy array and then to DataFrame
        data_val_X = pd.DataFrame(np.frombuffer(received_value, dtype=int).reshape(-1, 6), columns=['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'])
        #print("Pandas DataFrame: ", data_val_X)

        y_pridction = pridiction(clf,data_val_X)
        print(f"final y_pridction:  {y_pridction}")

        # Send y_prediction to RPI-2
        send_host = '192.168.2.6'  
        send_port = 12346         
        send_array(send_host, send_port, y_pridction)

        #close Server socket
        client_socket.close()

if __name__ == "__main__":
    start_server()
# %%
