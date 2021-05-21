import pandas as pd
from dfply import *
import pickle

demand_model = pickle.load(open("Demand_model.sav", 'rb'))
supply_model = pickle.load(open("Model_Supply.sav", 'rb'))
model= demand_model >> inner_join(supply_model, by='Date')
L=[]
N=model.shape
for i in range(N[0]):
	if model.loc[i,"Supply"]-model.loc[i,"Demand"]>0 :
		L.append("SELL")
	elif model.loc[i,"Supply"]- model.loc[i,"Demand"]<0 :
		L.append("BUY")
	else :
		L.append("FLAT")
model["decision"]=L
print(model)

