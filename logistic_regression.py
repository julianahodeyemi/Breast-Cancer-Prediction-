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
accu = cross_val_score(estimator=log_reg, X = X_train, y= y_train, cv=10)
print('Accuracy : {:.2f} %'.format(accu.mean()*100))
print('Accuracy : {:.2f} %'.format(accu.std()*100))

#%%
# Applying Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'penalty':['l1','l2']}]
grid_search = GridSearchCV(estimator = log_reg,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
# %%
