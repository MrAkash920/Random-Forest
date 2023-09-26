#!/usr/bin/env python
# coding: utf-8

# In[33]:


#import libarary
import numpy as np
import pandas as pd
#import dataset from iris dataset
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
np.random.seed(0)


# In[34]:


#create an object iris from load_iris
iris = load_iris()
#create a dataframe
df = pd.DataFrame(iris.data, columns= iris.feature_names)
df.head()


# In[35]:


#Adding new column for the species name
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()


# In[36]:


#creating test and train data
df['is_train']= np.random.uniform(0,1,len(df))<=.75
df.head()


# In[37]:


#creating dataframe with train rows and test rows
train, test =  df[df['is_train']== True], df[df['is_train']== False]
#Print the lenth of the Train and test dataframes
print("Length of the train dataframes:",len(train))
print("Length of the test dataframes:",len(test))


# In[38]:


#list of the features
features = df.columns[:4]
features


# In[39]:


#convert each spacies name into digits
y = pd.factorize(train['species'])[0]
y


# In[40]:


#Create a random forest classifier
rf = RandomForestClassifier(n_jobs=2, random_state=0)
#train the classifier
rf.fit(train[features],y)


# In[41]:


#test dataset
rf.predict(test[features])


# In[42]:


#viewing the predictive probabilities of the first 10 observation
rf.predict_proba(test[features])[0:10]


# In[43]:


#mapping the names for the plants for each predicted plant class
preds = iris.target_names[rf.predict(test[features])]
preds[0:20]


# In[44]:


#viewing the actual species 
test['species'].head()


# In[46]:


#creating confusion matrix
confusion_matrix = pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])
confusion_matrix


# In[47]:


import seaborn as sns
import matplotlib.pyplot as plt
# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Confusion Matrix')
plt.show()


# In[52]:


# Assuming you have your input data in a DataFrame
input_data = pd.DataFrame([[5.0, 3.6, 1.4, 2.0], [5.0, 3.6, 1.4, 2.0]], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

# Now, you can use these feature names for predictions
preds = iris.target_names[rf.predict(input_data)]
preds


# In[ ]:




