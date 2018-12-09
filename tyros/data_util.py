#!/usr/bin/python

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import numpy as np

def disp(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
    
def missings(df):
    disp(df.isnull().sum().sort_index() / len(df))

def split_vals(a,n): 
    return a[:n].copy(), a[n:].copy()

def get_train_val(df, y, n_valid):
    n_trn = len(df)-n_valid
    X_train, X_valid = split_vals(df, n_trn)
    y_train, y_valid = split_vals(y, n_trn)
    return X_train, X_valid, y_train, y_valid

def rmse(a, b):
    return np.sqrt(((a - b)**2).mean())

def print_score(m, X_train, X_valid, y_train, y_valid):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
    
def run(X_train, X_valid, y_train, y_valid, rf_args = {'n_jobs': -1}):
    m = RandomForestRegressor(**rf_args)
    m.fit(X_train, y_train)
    print_score(m, X_train, X_valid, y_train, y_valid)
    return m