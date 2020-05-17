#!/usr/bin/env python
# coding: utf-8

# In[27]:


from sklearn import datasets


# In[28]:


iris=datasets.load_iris()


# In[29]:


iris


# # 注意到iris文件是一个dict

# In[32]:


print(iris.keys())


# In[40]:


X=iris['data']


# In[41]:


X


# In[42]:


y=iris['target']


# In[43]:


y


# In[44]:


iris['target_names']


# In[45]:


import pandas as pd 


# In[46]:


df=pd.DataFrame(X,columns=iris['feature_names']) #columns补充名称


# In[47]:


df


# In[48]:


print(df.head())


# # Visual EDA

# In[60]:


A=pd.plotting.scatter_matrix(df,c=y,figsize=[10,10],s=150,marker='B')


# # Use Scikit-learn to Fit a Classifier分类器-KNN Before Train/Test Split

# In[64]:


from sklearn.neighbors import KNeighborsClassifier 


# In[66]:


knn=KNeighborsClassifier(n_neighbors=6)


# In[67]:


knn.fit(X,y)


# In[69]:


X.shape #X=iris['data']


# In[70]:


y.shape #y=iris['target']


# In[94]:


y_predict=knn.predict(X)


# In[103]:


knn.score(X,y) #still no test dateset for x, so need to create test dataset


# # Predict and print the label for the new data point X_new

# In[96]:


import numpy as np
X_new=np.random.randint(0,10,size=[3,4])


# In[97]:


prediction=knn.predict(X_new)


# In[101]:


print('Prediction by KNN before train/test split:{}'.format(prediction))


# # Use Scikit-learn to Fit a Classifier-KNN After Train/Test Split

# In[104]:


from sklearn.model_selection import train_test_split


# In[105]:


from sklearn.neighbors import KNeighborsClassifier 


# In[106]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)


# In[107]:


knn=KNeighborsClassifier(n_neighbors=9)


# In[108]:


knn.fit(X_train,y_train)


# In[109]:


y_predict=knn.predict(X_test)


# In[110]:


print('Prediction by KNN before train/test split:{}'.format(y_predict))


# In[111]:


knn.score(X_test,y_test)


# # Measure Model Performance & Choose a best K For Fit a classifier KNN

# In[168]:


from sklearn.model_selection import train_test_split


# In[169]:


from sklearn.neighbors import KNeighborsClassifier 


# In[170]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)


# In[171]:


neighbors=np.arange(1,25)


# In[172]:


Train_accuracy=np.empty(len(neighbors))


# In[173]:


Test_accuracy=np.empty(len(neighbors))


# In[174]:


import matplotlib.pyplot as plt


# In[175]:


# Loop 
for i,k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k)
    # Fit a classifier to the training data
    knn.fit(X_train,y_train)
    
    Train_accuracy[i]=knn.score(X_train,y_train)
    Test_accuracy[i]=knn.score(X_test,y_test)
    
# generate plot
plt.title('Getting view of best k')
plt.plot(neighbors,Train_accuracy,label='Training Accuracy',c='b')
plt.plot(neighbors,Test_accuracy,label='Testing Accuracy',c='r')
plt.legend()
plt.xlabel('k=Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




