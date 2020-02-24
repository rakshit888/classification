#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


os.chdir(r'C:\Users\HP\Desktop\PYTHON AND DATA SCIENCE - Copy')


# In[3]:


dataset=pd.read_csv('CREDIT_TRAIN_DATA.csv')


# In[4]:


dataset.isnull().sum()/len(dataset)*100


# In[5]:


dataset.info()


# In[6]:


import seaborn as sns
sns.boxplot(y='LoanAmount', data=dataset)


# In[7]:


dataset['Gender'] = dataset['Gender'].fillna('Male')


# In[8]:



dataset['Married'] = dataset['Married'].fillna('Yes')


# In[9]:


dataset['Dependents'] = dataset['Dependents'].fillna('0')


# In[10]:


dataset['Self_Employed'] = dataset['Self_Employed'].fillna('No')


# In[11]:


dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].median())


# In[12]:



dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].fillna(360)


# In[13]:



dataset['Credit_History'] = dataset['Credit_History'].fillna(1.0)


# In[14]:


dataset.isnull().sum()/len(dataset)*100


# In[15]:


dataset['Gender'] = dataset['Gender'].astype('category')
dataset['Gender'] = dataset['Gender'].cat.codes


# In[16]:


dataset['Married'] = dataset['Married'].astype('category')
dataset['Married'] = dataset['Married'].cat.codes


# In[17]:


dataset['Dependents'] = dataset['Dependents'].astype('category')
dataset['Dependents'] = dataset['Dependents'].cat.codes


# In[18]:


dataset['Education'] = dataset['Education'].astype('category')
dataset['Education'] = dataset['Education'].cat.codes


# In[19]:


dataset['Self_Employed'] = dataset['Self_Employed'].astype('category')
dataset['Self_Employed'] = dataset['Self_Employed'].cat.codes


# In[20]:


dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].astype('category')
dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].cat.codes


# In[21]:


dataset['Property_Area'] = dataset['Property_Area'].astype('category')
dataset['Property_Area'] = dataset['Property_Area'].cat.codes


# In[22]:


dataset['Loan_Status'] = dataset['Loan_Status'].astype('category')
dataset['Loan_Status'] = dataset['Loan_Status'].cat.codes


# In[23]:


dataset


# In[24]:



x = dataset.iloc[:,1:12].values


# In[25]:


y = dataset.iloc[:,-1].values


# In[26]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)


# In[27]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x,y)


# In[28]:


os.chdir(r'C:\Users\HP\Desktop\PYTHON AND DATA SCIENCE - Copy')


# In[29]:


test=pd.read_csv('CREDIT VALIDATE DATA.csv')


# In[30]:


test


# In[31]:


test.isnull().sum()/len(test)*100


# In[32]:


test['Gender'] = test['Gender'].fillna('Male')


# In[33]:


test['Gender'] = test['Gender'].astype('category')
test['Gender'] = test['Gender'].cat.codes


# In[34]:


test['Dependents'] = test['Dependents'].fillna('0')


# In[35]:


test['Dependents'] = test['Dependents'].astype('category')
test['Dependents'] = test['Dependents'].cat.codes


# In[36]:


test['Self_Employed'] = test['Self_Employed'].fillna('No')


# In[37]:


test['Self_Employed'] = test['Self_Employed'].astype('category')
test['Self_Employed'] = test['Self_Employed'].cat.codes


# In[38]:



test['LoanAmount'] = test['LoanAmount'].fillna(test['LoanAmount'].median())


# In[39]:


test['Loan_Amount_Term'] = test['Loan_Amount_Term'].fillna(360)


# In[40]:


test['Loan_Amount_Term'] = test['Loan_Amount_Term'].astype('category')
test['Loan_Amount_Term'] = test['Loan_Amount_Term'].cat.codes


# In[41]:


test['Credit_History'] = test['Credit_History'].fillna(1.0)


# In[42]:



test['Married'] = test['Married'].astype('category')
test['Married'] = test['Married'].cat.codes


# In[43]:


test['Education'] = test['Education'].astype('category')
test['Education'] = test['Education'].cat.codes


# In[44]:


test['outcome'] = test['outcome'].astype('category')
test['outcome'] = test['outcome'].cat.codes


# In[45]:



test['Property_Area'] = test['Property_Area'].astype('category')
test['Property_Area'] = test['Property_Area'].cat.codes


# In[46]:


test


# In[47]:


x_test = test.iloc[:,1:12].values


# In[48]:



y_test = test.iloc[:,-1].values


# In[49]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_test = sc_x.fit_transform(x_test)


# In[50]:


y_pred = logmodel.predict(x_test)


# In[51]:


y_pred


# In[52]:


from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test, y_pred)


# In[53]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # K-Nearest Neighbour

# In[54]:


from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', p=2)
classifier_knn.fit(x,y)


# In[55]:



y_pred = classifier_knn.predict(x_test)


# In[56]:


cm=confusion_matrix(y_test,y_pred)


# In[57]:


cm


# In[58]:



(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # Naive Bayes

# In[59]:


from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(x, y)


# In[61]:



y_pred = classifier_nb.predict(x_test)


# In[62]:


cm=confusion_matrix(y_test,y_pred)


# In[63]:



(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # Support Vector Machine - Linear Kernel1

# In[64]:


from sklearn.svm import SVC
classifier_svm_linear = SVC(kernel='linear')
classifier_svm_linear.fit(x, y)


# In[65]:



y_pred = classifier_svm_linear.predict(x_test)


# In[66]:


cm=confusion_matrix(y_test,y_pred)


# In[67]:



(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # Support Vector Machine - Sigmoid Kernel¶

# In[68]:


from sklearn.svm import SVC
classifier_svm_sigmoid = SVC(kernel='sigmoid')
classifier_svm_sigmoid.fit(x, y)


# In[69]:



y_pred = classifier_svm_sigmoid.predict(x_test)


# In[70]:


cm=confusion_matrix(y_test,y_pred)


# In[71]:



(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # Support Vector Machine - Radial Basis Function Kernel

# In[72]:


from sklearn.svm import SVC
classifier_svm_rbf = SVC(kernel='rbf')
classifier_svm_rbf.fit(x, y)


# In[73]:



y_pred = classifier_svm_rbf.predict(x_test)


# In[74]:



cm=confusion_matrix(y_test,y_pred)


# In[75]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # Support Vector Machine - Polynomial Function Kernel
# 

# In[76]:


from sklearn.svm import SVC
classifier_svm_poly = SVC(kernel='poly')
classifier_svm_poly.fit(x, y)


# In[77]:



y_pred = classifier_svm_poly.predict(x_test)


# In[78]:



(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # Decision Tree
# 

# In[79]:


from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion='entropy')
classifier_dt.fit(x, y)


# In[80]:


y_pred = classifier_dt.predict(x_test)


# In[81]:


cm=confusion_matrix(y_test, y_pred)


# In[82]:



(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # Random Forest

# In[83]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=3, criterion='entropy')
classifier_rf.fit(x, y)


# In[84]:


y_pred = classifier_rf.predict(x_test)


# In[85]:


cm=confusion_matrix(y_test, y_pred)


# In[86]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# In[87]:


validation=pd.read_csv('CREDIT_TEST_DATA.csv')


# In[88]:


validation


# In[89]:



validation.isnull().sum()/len(validation)*100


# In[90]:


validation.groupby('Credit_History').size()


# In[91]:



validation['Gender'] = validation['Gender'].fillna('Male')


# In[92]:


validation['Gender'] = validation['Gender'].astype('category')
validation['Gender'] = validation['Gender'].cat.codes


# In[93]:



validation['Dependents'] = validation['Dependents'].fillna('0')


# In[94]:


validation['Dependents'] = validation['Dependents'].astype('category')
validation['Dependents'] = validation['Dependents'].cat.codes


# In[95]:


validation['Self_Employed'] = validation['Self_Employed'].fillna('No')


# In[97]:


validation['Self_Employed'] = validation['Self_Employed'].astype('category')
validation['Self_Employed'] = validation['Self_Employed'].cat.codes


# In[98]:


import seaborn as sns
sns.boxplot(y='LoanAmount', data=validation)


# In[99]:


validation['LoanAmount'] = validation['LoanAmount'].fillna(validation['LoanAmount'].median())


# In[100]:


validation['Loan_Amount_Term'] = validation['Loan_Amount_Term'].fillna(360)


# In[101]:



validation['Credit_History'] = validation['Credit_History'].fillna(1.0)


# In[102]:


validation['Married'] = validation['Married'].astype('category')
validation['Married'] = validation['Married'].cat.codes


# In[103]:


validation['Education'] = validation['Education'].astype('category')
validation['Education'] = validation['Education'].cat.codes


# In[104]:


validation['Property_Area'] = validation['Property_Area'].astype('category')
validation['Property_Area'] = validation['Property_Area'].cat.codes


# In[105]:


x_val=validation.iloc[:,1:12].values


# In[106]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_val = sc_x.fit_transform(x_val)


# # Applying model in the validation data

# In[107]:



y_pred = logmodel.predict(x_val)
y_pred


# In[108]:


y_pred = classifier_knn.predict(x_val)
y_pred


# In[109]:


y_pred = classifier_nb.predict(x_val)
y_pred


# In[110]:



y_pred = classifier_svm_linear.predict(x_val)
y_pred


# In[111]:


y_pred = classifier_svm_sigmoid.predict(x_val)
y_pred


# In[112]:


y_pred = classifier_svm_rbf.predict(x_val)
y_pred


# In[113]:


y_pred = classifier_svm_poly.predict(x_val)
y_pred


# In[114]:


y_pred = classifier_dt.predict(x_val)
y_pred


# In[115]:


y_pred = classifier_rf.predict(x_val)
y_pred


# In[ ]:




