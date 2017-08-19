# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os

def main():
    
    # Change working directory to the file's location
    os.chdir(os.path.dirname(__file__))
    
    #import traing data as a pandas data frame, ignore first 3 cols as they are not features
    df = pd.read_csv('.\\Data\\numerai_training_data.csv', header=0, usecols=list(range(3, 25)))

    X = df.iloc[:,0:20].as_matrix()
    Y = df.iloc[:,21].as_matrix()
    
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]