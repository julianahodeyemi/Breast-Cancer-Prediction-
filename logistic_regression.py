#%%
#Import libraries
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

#%%
#Import dataset
data= pd.read_csv('breast_cancer.csv')
data.head()

#%%
#EDA with profile report to understand the data more
dataset_eda = ProfileReport(data)
dataset_eda.to_notebook_iframe()

# %%
X= data.iloc[:,1:-1].values
y= data.iloc[:,-1].values
# %%
#Split into test and train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#%%
#Standardization of the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# %%
#Training logistic regression model on training set
log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X_train, y_train)

# %%
#Predict test set result
y_test_pred = log_reg.predict(X_test)

# %%
#Confusion Matrix
cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)
accuracy_score(y_test, y_test_pred)
# %%
#Accuracy with k-fold cross validation
accu = cross_val_score(estimatoer=log_reg, x = X_train, y= y_train, cv=10)
print('Accuracy : {:.2f} %'.format(accu.mean()*100))
print('Accuracy : {:.2f} %'.format(accu.mean()*100))