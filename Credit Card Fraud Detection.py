
# coding: utf-8

# In[11]:


#import sys



# In[12]:


#importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


#Reading the csv file
data=pd.read_csv('/home/aman/Desktop/creditcard.csv')


# In[14]:


print(data.shape)


# In[15]:


print(data.describe())


# In[16]:


#Taking out some part of data
data=data.sample(frac=0.1,random_state=1)
print(data.shape)


# In[20]:


data.hist(figsize=(20,20))
plt.show()


# In[21]:


Fraud=data[data['Class'] == 1]
valid=data[data['Class'] == 0]


outlier_fraction=len(Fraud)/float(len(valid))
print(outlier_fraction)


print('Fraud Cases: {}'.format(len(Fraud)))
print('Valid Cases: {}'.format(len(valid)))


# In[23]:


#Corelation matrix
corrmat=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()


# In[25]:


#Get all the columns from the Dataframe
columns=data.columns.tolist()


#Filter the columns

columns=[c for c in columns if c not in ["Class"]]



#storing the variable that we have to work on
target="Class"
X=data[columns]
Y=data[target]


#Print the shape X and y
print(X.shape)
print(Y.shape)


# In[27]:


from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# In[30]:


#define a random state
state=1


#outlier detection method

classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),contamination=outlier_fraction,random_state=state),
    
    "Local Outlier Factor":LocalOutlierFactor(
    n_neighbors=20,
    contamination=outlier_fraction)
}
    


# In[33]:


#Fit the model

n_outliers=len(Fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):
    #fit the data
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred = clf.predict(X)
        
        
    #Reshape
    
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    
    
    n_errors = (y_pred != Y).sum()
    
    
    #Run classification metrics
    
    print('{}: {}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))

