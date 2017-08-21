# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import linear_model, svm
from xgboost import XGBClassifier
import os

def main():
    
    # Change working directory to the file's location
    os.chdir(os.path.dirname(__file__))
    
    #import traing data as a pandas data frame, ignore first 3 cols as they are not features
    df = pd.read_csv('.\\Data\\numerai_training_data.csv', header=0, usecols=list(range(3, 25)))

    #split data into features and target
    X = df.iloc[:,0:21].as_matrix()
    y = df.iloc[:,21].as_matrix()        
        
    lr = linear_model.LogisticRegression()
    lrScores = cross_val_score(lr, X, y, cv=10, scoring = 'f1')
    sum(lrScores) / len(lrScores)
    
    xgb = XGBClassifier()
    xgbScores = cross_val_score(xgb, X, y, cv=10, scoring = 'f1')
    sum(xgbScores) / len(xgbScores)
    
    clf = svm.SVR()
    svmScores = cross_val_score(clf, X, y, cv=10)
    sum(svmScores) / len(svmScores)