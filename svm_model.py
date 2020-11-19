# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 00:05:03 2020

@author: sakshar5068
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import smote_variants as sv

data_file = ['transfusion.csv','haberman.csv','breast-cancer-wisconsin.csv']
file_no = 1

# loading the dataset
trans_data = pd.read_csv(data_file[file_no]).to_numpy()
X_data, Y_data = trans_data[:, 0:-1], trans_data[:, -1]
print("-----------without oversampling--------------")
print(X_data.shape, Y_data.shape)

# standardizing data and declaring k-fold cross validation
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# setting gemma value for RBF kernel
g = .01

# without oversampling
acc, prec, f1 = 0.0, 0.0, 0.0
for train_index, test_index in skf.split(X_data, Y_data):
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = Y_data[train_index], Y_data[test_index]
    #print(X_train.shape)
    #print(X_test.shape)
    clf = svm.SVC(kernel='rbf', random_state=0, gamma=g, C=1) # using RBF kernel
    #clf = svm.SVC(kernel='linear') # using linear kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = acc +  metrics.accuracy_score(y_test, y_pred)
    prec = prec + metrics.precision_score(y_test, y_pred)
    f1 = f1 + metrics.f1_score(y_test, y_pred)
acc, prec, f1 = acc/5.0, prec/5.0, f1/5.0
print("Accuracy:", acc)
print("Precision:", prec)
print("F1 Score:", f1)

# using basic SMOTE oversampling
print("-----------using SMOTE--------------")
oversampler = sv.SMOTE()
acc, prec, f1 = 0.0, 0.0, 0.0
for train_index, test_index in skf.split(X_data, Y_data):
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = Y_data[train_index], Y_data[test_index]
    X_samp, y_samp = oversampler.sample(X_train, y_train)
    #print(X_samp.shape, y_samp.shape)
    #print(X_test.shape)
    clf = svm.SVC(kernel='rbf', random_state=0, gamma=g, C=1) # using RBF kernel
    #clf = svm.SVC(kernel='linear') # using linear kernel
    clf.fit(X_samp, y_samp)
    y_pred = clf.predict(X_test)
    acc = acc +  metrics.accuracy_score(y_test, y_pred)
    prec = prec + metrics.precision_score(y_test, y_pred)
    f1 = f1 + metrics.f1_score(y_test, y_pred)
acc, prec, f1 = acc/5.0, prec/5.0, f1/5.0
print("Accuracy:", acc)
print("Precision:", prec)
print("F1 Score:", f1)

# using CURE_SMOTE oversampling
print("-----------using CURE_SMOTE--------------")
oversampler2 = sv.CURE_SMOTE()
acc, prec, f1 = 0.0, 0.0, 0.0
for train_index, test_index in skf.split(X_data, Y_data):
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = Y_data[train_index], Y_data[test_index]
    X_samp, y_samp = oversampler2.sample(X_train, y_train)
    #print(X_samp.shape, y_samp.shape)
    #print(X_test.shape)
    clf = svm.SVC(kernel='rbf', random_state=0, gamma=g, C=1) # using RBF kernel
    #clf = svm.SVC(kernel='linear') # using linear kernel
    clf.fit(X_samp, y_samp)
    y_pred = clf.predict(X_test)
    acc = acc +  metrics.accuracy_score(y_test, y_pred)
    prec = prec + metrics.precision_score(y_test, y_pred)
    f1 = f1 + metrics.f1_score(y_test, y_pred)
acc, prec, f1 = acc/5.0, prec/5.0, f1/5.0
print("Accuracy:", acc)
print("Precision:", prec)
print("F1 Score:", f1)


