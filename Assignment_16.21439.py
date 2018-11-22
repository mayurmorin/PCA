
# coding: utf-8

# ## In this assignment I have to transform iris data into 3 dimensions and plot a 3d chart with transformed dimensions and color each data point with specific class.

# ### Importing Modules

# In[1]:


import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition, datasets
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# ### Loading Data

# In[2]:


#Load and return the iris dataset
iris = datasets.load_iris()


# ### Data Exploration/Analysis

# In[3]:


#iris is Dictionary-like object with following keys
iris.keys()


# In[4]:


#Prints iris dataset description
print(iris.DESCR)


# In[5]:


#Prints iris dataset 'data' values
print(iris.data)


# In[6]:


#Prints iris dataset 'target' values 
print(iris.target)


# In[7]:


#Prints iris dataset 'target_names' values
print(iris.target_names)


# In[8]:


#Prints iris dataset 'feature_names' values
print(iris.feature_names)


# In[9]:


#Prints shape of the iris dataset 'data' values
print(iris.data.shape)


# In[10]:


#Prints shape of the iris dataset 'target' values
print(iris.target.shape)


# ### Data Visualization

# In[11]:


#Creating iris_df dataframe with custom column names
custom_column_name = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
iris_df = pd.DataFrame(iris.data, columns=custom_column_name)


# In[12]:


#Prints shape of the iris_df dataframe
print(iris_df.shape)


# In[13]:


iris_df.head() #Returns the first 5 rows of iris_df dataframe


# In[14]:


#Box and whisker plots
iris_df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False,figsize=(16,9))
plt.show()


# In[15]:


#Histograms
iris_df.hist(figsize=(16,9))
plt.show()


# In[16]:


#Scatter plot matrix
scatter_matrix(iris_df, figsize=(16,9))
plt.show()


# ### Transforming iris data into 3 dimensions

# In[17]:


#Using sklearn.decomposition.PCA for 3 components and transforming the dimension of iris.data values
X_reduced = PCA(n_components=3).fit_transform(iris.data)
X_reduced


# In[18]:


#Assigning iris.target values to variable y
y = iris.target
y


# In[19]:


#Plots a 3d chart with transformed dimensions and color each data point with specific class
fig = plt.figure(figsize=(16, 9))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Three PCA directions of transformed iris data")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")
plt.show()

