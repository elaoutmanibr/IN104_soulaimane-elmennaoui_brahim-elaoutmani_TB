
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
	L_lr  = []
	L_X = []
   	#logistic regression method
	for key,df in d.items():
		df = df >> inner_join(price_data,by='gasDayStartedOn')
		df = df.dropna()
		x = df >> select(X.LNW,X.FSW1,X.FSW2,X.SAS_GPL,X.SAS_TTF)
		Y= df['Net Withdrawal_binary'].values
		
		x_train, x_test, y_train, y_test = train_test_split(x, Y, random_state=1)
		lr = LogisticRegression()
		lr.fit(x_train, y_train)
		y_pred = lr.predict(x_test)
		cm = confusion_matrix(y_test, y_pred)
		probs=lr.predict_proba(x_test)[:, 1]
		L_lr.append(lr)
		L_X.append(x)
		dic1[key] = {'recall': recall_score(y_test, y_pred),'neg_recall': cm[1,1]/(cm[0,1] + cm[1,1]),'confusion': cm,'precision': precision_score(y_test,y_pred),'neg_precision':cm[1,1]/cm.sum(axis=1)[1],'roc': roc_auc_score(y_test, probs),'class_mod': lr } 
		#random tree method
		model=RandomForestClassifier(n_estimators=100, bootstrap = True,max_features = 'sqrt')
		model.fit(x_test,y_test)
		y_pred_tree=model.predict(x_test)
		cm_tree=confusion_matrix(y_test,y_pred_tree)
		probs_tree=model.predict_proba(x_test)[:, 1]
		dic2[key] = {'recall': recall_score(y_test, y_pred_tree), 'neg_recall': cm_tree[1,1]/(cm_tree[0,1] + cm_tree[1,1]), 'confusion': cm_tree, 'precision': precision_score(y_test, y_pred_tree), 'neg_precision':cm_tree[1,1]/cm_tree.sum(axis=1)[1], 'roc': roc_auc_score(y_test, probs_tree),'class_mod': model }
		
	pickle.dump(L_lr, open("LogReg.sav", 'wb'))
	pickle.dump(L_X, open("X.sav", 'wb'))

def  regression(d,price_data):
	dic={}
	L_r = []
	for key,df1 in d.items():
		df1 = df1 >> inner_join(price_data,by='gasDayStartedOn')
		df1 = df1.dropna()
		df=df1.loc[df1['Net Withdrawal_binary']==1]
		x = df >> select(X.LNW,X.FSW1,X.FSW2,X.SAS_GPL,X.SAS_TTF)
		y= df['NW'].values
		x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
		regressor = LinearRegression()  
		regressor.fit(x_train, y_train) #training the algorithm
		y_pred = regressor.predict(x_test)
		pickle.dump(L_r, open("RegMD.sav", 'wb'))
		corr, _ = pearsonr(y_test,y_pred)
		rmse = np.sqrt(mean_squared_error(y_test,y_pred))
		nrmse = rmse/(np.max(y_test) - np.min(y_test))
		anrmse = rmse/np.mean(y_test)
		dic[key]= {'r2': r2_score(y_test, y_pred), 'rmse': rmse, 'nrmse': nrmse, 'anrmse': anrmse, 'cor': corr, 'l_reg': regressor}
	



            

d = open_sheet()
build_model(d)
priceData = import_data()
regression(d, priceData)








