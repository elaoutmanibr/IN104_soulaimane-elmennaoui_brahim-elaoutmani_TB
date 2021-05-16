
import pandas as pd
import os
from dfply import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import numpy as np



def open_sheet(filename="storage_data.xlsx"):
    d = pd.read_excel(filename,sheet_name = None)
    return d


def build_model(d): 
    for df in d.values():
        df['NW'] = df['withdrawal']-df['injection']
        df['LNW'] = df['NW'].shift()
        NWB=[]
        for v in df['NW']:
            if v>0:
                NWB.append(1)
            else: 
                NWB.append(0)
        df['Net Withdrawal_binary'] = NWB

        FSW1=[]
        FSW2=[]
        for v in df['full']:
            FSW1.append(max(v-45,0))
            FSW2.append(max(45-v,0))
        df["FSW1"] = FSW1
        df["FSW2"] = FSW2
    return


'''
def add_NW(d):
    for df in d.values():
        df['NW'] = df['withdrawal']-df['injection']
    return 


def add_LNW(d):
    for df in d.values():
        df['LNW'] = df['NW'].shift()
    return

def add_NW_binary(d):
    for df in d.values():
        df['Net Withdrawal_binary'] = int(df['NW']>0)
    return

def add_FSW(d):
    for df in d.values():
        df["FSW1"] = max(df['full']-45, 0)
        df["FSW2"] = max(45-df['full'], 0)
    return
'''
def import_data(filename="price_data.csv"):
    df = pd.read_csv(filename,sep=';')
    
    #df >>= rename(gasDayStartedOn='Date')
    df = df >> mutate(gasDayStartedOn = pd.to_datetime(df['Date']))
    return df

def classification(d,price_data):
    dic1 = {}
    dic2 = {}

    for key,df in d.items():
        df = df >> inner_join(price_data,by='gasDayStartedOn')

        x = df >> select(X.LNW,X.FSW1,X.FSW2,X.SAS_GPL,X.SAS_TTF)
        y = df['Net Withdrawal_binary']

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        dic1[key] = {'recall': recall_score(y_test, y_pred),
                    'neg_recall': cm[1,1]/(cm[0,1] + cm[1,1]),
                    'confusion': cm,
                    'precision': precision_score(y_test, y_pred),
                    'neg_precision':cm[1,1]/cm.sum(axis=1)[1],
                    'roc': roc_auc_score(y_test, probs), #lr.probs()...
                    'class_mod': "the logistic regression"}

d = open_sheet()
build_model(d)
priceData = import_data()
classification(d, priceData)








