import pandas as pd
import os
from dfply import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier


import matplotlib.pyplot as plt
import numpy as np

import pickle

date = pickle.load(open("Date.sav","rb"))
logReg = pickle.load(open("LogReg.sav","rb"))
X = pickle.load(open("X.sav","rb"))
regML = pickle.load(open("RegMD.sav","rb"))
i = 0 # X and logReg have the same length
allStorages =[]

def supply_df(date_list = date, LR_list = logReg, LinR_list = regML, X_list = X):
    i = 0 # X and logReg have the same length
    allStorages =[]
    for LR in LR_list:
        x = X_list[i]
        Y = LR.predict(x)
        NW = LinR_list[i].predict(x)
        x['y'] = Y
        x['NW_f'] = NW
        x['Date'] = date_list[i]
        x=x.loc[x['y']==1]
        i += 1
        allStorages.append(x)

    L =[] # list of all possible dates
    for D in date_list :
        for i in D: 
            if i not in L :
                L.append(i)

    tab = {}

    for d in L:
        supply = 0
        for S in allStorages:
            d_col = S['Date'].values
            nw_col = S['NW_f'].values
            N = len(d_col)
            for i in range(N):
                if d_col[i] == d :
                    supply += nw_col[i]
        
        tab[d] = supply
    df = pd.DataFrame(list(tab.items()),columns=['Date','Supply'])
    return df

S_df = supply_df()

pickle.dump(S_df, open("Model_Supply.sav", 'wb'))




