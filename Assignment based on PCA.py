#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[3]:


df= pd.read_csv("PCA_practice_dataset.csv")


# In[4]:


df.head()


# In[5]:


x=df.to_numpy()


# In[6]:


x.shape


# In[7]:


scaler= StandardScaler()
x=scaler.fit_transform(x)


# In[9]:


pca= PCA()
x=pca.fit_transform(x)
cumalitive_variance = np.cumsum(pca.explained_variance_ratio_)*100
thresholds=[i for i in range(90,97+1,1)]
components=[np.argmax(cumalitive_variance>threshold)for threshold in thresholds]
for component, threshold in zip(components, thresholds):
        print("Components required for ", threshold, "% threshold are :", component)


# In[10]:


plt.plot(components, range(90,97+1,1),'ro-', linewidth=2)
plt.title('Screen Plot')
plt.xlabel('Principle Component')
plt.ylabel('Threshold in %')
plt.show


# In[ ]:




