#!/usr/bin/env python
# coding: utf-8

# # ANN Based Fault Detection
# 
# In this code I have implemented an Artificial Neural Network based power system line faults detection. In this notebook, a  Multi-layer perceptron (MLP) Classifier is implemented which uses Backpropagation. The data used is a time series data which was generated in MATLAb. Two target values 0 and 1 are used to create binary classes. When 0 is identified then there is no fault and 1 shows fault is detected. The data consist of different features and only six features (3-phase current and voltage measurements are used to train the MLP model). The code can be divided into following steps
# 
# 1. Data is read, preprocessed, and statistically analyzed.
# 
# 2. After pre-processing the data is splitted into test, train, and predict datasets
# 
# 3. MLP classifier model is trained with trainning dataset, tested, and predicted with seperated data which was not used in training process.
# 
# 4. Results of MLP classifier
# 
# 5. Data Visualization
# 

# ### Reading Data

# In[1]:


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


# In[2]:


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


clf = MLPClassifier(activation='relu', hidden_layer_sizes=(6,5,3),max_iter=500)
clf.fit(X_train, y_train)
print("Training Score:", clf.score(X_train,y_train))
print("Test Score:", clf.score(X_test,y_test))
scores = cross_val_score(clf, X_test, y_test, cv=5)
print("CV Score:", np.mean(scores))
metrics.ConfusionMatrixDisplay.from_estimator(clf,X_test,y_test)
plt.show()
svc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)


# ### Calculating accuracy of predit dataset and plotting confusion matrix

# In[9]:


y_pred=clf.predict(data_val_X)
print("Validation Score:",clf.score(data_val_X,data_val_Y) )
scores = cross_val_score(clf, data_val_X,data_val_Y, cv=5)
print("Cross validation Score:", np.mean(scores))
metrics.ConfusionMatrixDisplay.from_estimator(clf,data_val_X,data_val_Y)
plt.show()
svc_disp = RocCurveDisplay.from_estimator(clf, data_val_X, data_val_Y)


# ### Precision and accuracy of MLP model

# In[10]:


print(classification_report(data_val_Y, y_pred))


# ## Data Visualization in 2D and 3D

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=2)
Xreduced = pca.fit_transform(X_train)
XR_test = pca.fit_transform(X_test)


# In[14]:


def make_meshgrid(x, y, h=1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

clf = clf.fit(Xreduced, y_train)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of RBF SVC ')

# Set-up grid for plotting.
X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.figure(figsize=(10, 8))

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Decison surface using the PCA transformed/projected features')
#ax.legend()
plt.show()


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=3)
Xreduced = pca.fit_transform(X_train)
XR_test = pca.fit_transform(X_test)
XR_predict = pca.fit_transform(data_val_X)

X0, X1, X2 = Xreduced[:, 0], Xreduced[:, 1],Xreduced[:, 2]


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'widget')

ax = plt.figure().add_subplot(projection='3d')

ax.plot(X0, X1,X2, zdir='z', label='curve in (x, y)')

ax.scatter(X0, X1, X2, zdir='y',c=y_train,label='points in (x, z)')

# Make legend, set axes limits and labels
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.view_init(elev=20., azim=-35)

plt.show()


# In[17]:


# libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(X0,X1,X2, cmap=plt.cm.viridis, linewidth=0.2)
surf=ax.plot_trisurf(X0,X1,X2, cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
plt.show()


# In[ ]:




