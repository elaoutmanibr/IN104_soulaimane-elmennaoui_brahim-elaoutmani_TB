import pandas as pd
from dfply import *
import pickle


def  Decision(a="Demand_model.sav",b="Model_Supply.sav",c="real_d.sav",d="real_s.sav"):
	demand_model = pickle.load(open(a, 'rb'))
	supply_model = pickle.load(open(b, 'rb'))
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


	#model.to_csv("Final model",index=False,sep=';')

	real_demand = pickle.load(open(c, 'rb'))
	real_supply = pickle.load(open(d, 'rb'))

	model= model >> inner_join(real_demand, by='Date')	
	model= model >> inner_join(real_supply, by='Date')
	print(model)
	B=[]
	N=model.shape
	for i in range(N[0]):
		if model.loc[i,"real_supply"]-model.loc[i,"real_demand"]>0 :
			B.append("SELL")
		elif model.loc[i,"real_supply"]- model.loc[i,"real_demand"]<0 :
			B.append("BUY")
		else :
			B.append("FLAT")
	model["real_decision"]=B
	print(model)

#model.to_csv("Final model.csv",index=False,sep=';')
	i=0
	a=0
	N=model.shape
	for j in range(N[0]):
		if model.loc[i,"decision"]== model.loc[i,"real_decision"] :
			a+=1
		else :
			i+=1
	print(a)
	print(i)


Decision()