# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:14:35 2020

@author: diggee
"""

#%% importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% reading data files

full_train_data = pd.read_csv('train.csv')
full_test_data = pd.read_csv('test.csv')

#%% preprocessing data

print(full_test_data.isnull().sum().sum())
print(full_train_data.isnull().sum().sum())
# there is no missing data!

print((full_train_data.dtypes == object).any())
print((full_test_data.dtypes == object).any())
# there is no categorical data!

X = full_train_data.drop('label', axis = 1)
y = full_train_data['label']
X_valid = full_test_data

#%% classification models

def scaled_data(X_train, X_test, X_valid):
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    X_valid = scaler_X.transform(X_valid)
    return X_train, X_test, X_valid, scaler_X

def grid_search(parameters, classifier, criteria, X_train, y_train):
    from sklearn. model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = criteria, cv = 5)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    print(best_parameters)
    return best_accuracy, best_parameters

def classification(X_train, X_test, X_valid, y_train, choice):
    # reg_models = {1:'logistic', 2:'KNN', 3:'decision tree', 4:'random forest', 5:'SVR', 6:'XG Boost', 7:'NB'}
    if choice == 1:
        from sklearn.linear_model import LogisticRegression
        parameters = {'penalty':['l1','l2'], 'C':[0.01, 0.1, 1, 10]}
        _, best_params = grid_search(parameters, LogisticRegression(), 'accuracy', X_train, y_train)
        classifier = LogisticRegression(penalty = best_params['penalty'], C = best_params['C'])

    elif choice == 2:
        from sklearn.neighbors import KNeighborsClassifier
        parameters = {'n_neighbors':[3,4,5,6,7], 'weights':['uniform','distance']}
        _, best_param = grid_search(parameters, KNeighborsClassifier(), 'accuracy', X_train, y_train)
        classifier = KNeighborsClassifier(n_neighbors = best_param['n_neighbors'], weights = best_param['weights'])

    elif choice == 3:
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier()

    elif choice == 4:
        from sklearn.ensemble import RandomForestClassifier
        parameters = {'n_estimators':[10, 50, 100, 200, 400, 500]}
        _, best_param = grid_search(parameters, RandomForestClassifier(), 'accuracy', X_train, y_train)
        classifier = RandomForestClassifier(n_estimators = best_param['n_estimators'])

    elif choice == 5:
        from sklearn.svm import SVC
        X_train, X_test, X_valid, scaler_X = scaled_data(X_train, X_test, X_valid)
        parameters = [{'C': [0.01, 0.1 , 1], 'kernel': ['rbf', 'sigmoid'], 'gamma': [0.01, 0.05, 0.1, 0.5]}]
        _, best_param = grid_search(parameters, SVC(), 'accuracy', X_train, y_train)
        classifier = SVC(kernel = best_param['kernel'], C = best_param['C'], gamma = best_param['gamma'])

    elif choice == 6:
        from xgboost import XGBClassifier
        parameters = [{'n_estimators':[100, 500, 1000], 'learning_rate':[0.01, 0.05, 0.1]}]
        _, best_param = grid_search(parameters, XGBClassifier(), 'accuracy', X_train, y_train)
        classifier = XGBClassifier(n_estimators = best_param['n_estimators'], learning_rate = best_param['learning_rate'])
        
    else:
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()

    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    return y_pred, classifier, X_valid

#%% evaluating prediction quality

def pred_quality(y_pred, y_test):
    from sklearn.metrics import confusion_matrix, classification_report
    print('Classification report:-\n')
    print(classification_report(y_test, y_pred))
    print('\nConfusion matrix:-\n')
    print(confusion_matrix(y_test, y_pred))

#%% evalating the model on the test data and exporting it to csv file
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1 if X.shape[0]<=100000 else 10000)
y_pred, classifier, X_valid = classification(X_train, X_test, X_valid, y_train, choice = 6)
pred_quality(y_pred, y_test)
      
y_valid_pred = classifier.predict(X_valid)
df = pd.DataFrame({'ImageId':full_test_data.index+1, 'Label':y_valid_pred})
df.to_csv('prediction.csv', index = False)