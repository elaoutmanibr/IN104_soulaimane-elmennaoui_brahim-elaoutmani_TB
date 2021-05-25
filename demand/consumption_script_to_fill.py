
import pandas as pd
import pickle 	
import os
from dfply import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#This function sets the working directory
def set_wd(wd):
    os.chdir(wd)

#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_csv(f_name = "DE.csv", delimiter = ";", plot = True):
    df = pd.read_csv(f_name,sep=delimiter)
    #df >>= rename(Date='Date (CET)')# we create another column named "Date" ? NO
    df >>= mutate(Date=pd.to_datetime(df['Date (CET)']))
    if plot: 
        dfig, axes = plt.subplots(nrows=3, ncols=1)
        df.plot(x='Date (CET)', y='LDZ', ax=axes[0])
        df.plot(x='Date (CET)', y='Actual', ax=axes[1])
        df.plot(x='Date (CET)', y='Normal', ax=axes[2])
        plt.show()
    return df

#This function creates a scatter plot given a DataFrame and an x and y column
def scatter_plot(dataframe = "DE.csv",delimiter = ";", x = "Actual", y = "LDZ", col = "red"):
    df = pd.read_csv(dataframe,sep=delimiter)
    df.plot.scatter(x=x,y=y,c=col)
    plt.show() 

#This function is the sigmoid function for gas consumption as a function of temperature
def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))

#The following function takes sigmoid parameters values and a temperature vector as input and plots the sigmoid, can compare it with the actual values
def consumption_sigmoid(t, real_conso, a = 500, b = -25, c = 2, d = 100, plot = True):
    h_hat = np.empty(len(t))
    h_hat = h(t, a, b, c, d)

    if plot:
        plt.plot(t,h_hat,'r')
        #if real_conso is not None you plot it as well
        if not isinstance(real_conso, type(None)):
            plt.plot(t,real_conso,'b.')
            if(len(t) != len(real_conso)):
                print("Difference in length between Temperature and Real Consumption vectors")
        # add title and legend and show plot
        plt.xlabel("Temperature")
        #plt.legend(handles=[red_patch,blue_patch],labels=["","Real Consumption"])
        plt.show()
        
    return h_hat

#The following function gets the fit metrics list between 2 sigmoids
def get_fit_metrics(h_hat, real_conso):
    if(len(h_hat) != len(real_conso)):
        print("Difference in length between Fit and Real Consumption vectors:\t",abs(len(h_hat)-len(real_conso)))
        return
    else:
        print("Same length ! ")
        corr, _ = pearsonr(h_hat, real_conso)
        rmse = np.sqrt(mean_squared_error(h_hat,real_conso))
        nrmse = rmse/(np.max(real_conso) - np.min(real_conso))
        anrmse = rmse/np.mean(real_conso)
        return [corr,rmse,nrmse,anrmse]

#The following class is the consumption class it takes sigmoid parameters as well as a temperature as input
class consumption:
    #Initialize class
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    #calculate the consumption given a temperature
    def get_consumption(self, temperature):
        return (self.d+self.a/(1+(self.b/(temperature-40))**self.c))


    #get the sigmoid considering a temperature between -40 and 39, use the function consumption_sigmoid above
    def sigmoid(self, p):
        t = np.arange (-40,40,1) # 39 is included !
        return consumption_sigmoid(t, None, plot=p)
    
    #This is what the class print if you use the print function on it
    def __str__(self):
        t = "consumption coeff initialized"
        return t

#The following class optimizes the parameters of the sigmoid and returns an object of class consumption
class optimize_sigmoid:
    #Initialize guess values that are common to all instances of the class
    __guess_a, __guess_b, __guess_c, __guess_d = 500, -25, 2, 100
    #g =[500, -25, 2, 100]

    def __init__(self, f):
        if isinstance(f, pd.DataFrame):
            if 'Actual' and 'LDZ' in f.columns:
                self.__f = f
            else:
                print("Class not initialized since f does not contain Actual and LDZ column names")
        else:
            print("Class not initialized since f is not a DataFrame")

    #optimize and return metrics use functions h, consumption_sigmoid defined above as well as get_fit_metrics
    def optimize(self):
        if self.__f is not None:
            self.__f = self.__f.dropna()
            t = self.__f['Actual'].values
            real = self.__f['LDZ'].values
            g = [self.__guess_a,self.__guess_b,self.__guess_c,self.__guess_d]
            
            self.__coef, self.__cov = curve_fit(h,t,real,g)
            G = [self.__coef[0],self.__coef[1],self.__coef[2],self.__coef[3]]
            s = consumption_sigmoid(t, real,
             a=self.__coef[0], 
             b=self.__coef[1],
             c=self.__coef[2],
             d=self.__coef[3],
             plot = True)
            
            self.__corr, self.__rmse, self.__nrmse, self.__anrmse = get_fit_metrics(s, self.__f['LDZ'])
            return (G)
        else:
            print("Class not initialized")

    #this function returns the fit metrics calculated above
    def fit_metrics(self):
        if  self.__coef is not None:
            return [self.__corr, self.__rmse, self.__nrmse, self.__anrmse]
        else:
            print("optimize method is not yet run")

    #This function creates the class consumption
    def create_consumption(self):
        if  self.__coef is not None:
            return consumption(self.__coef[0],self.__coef[1],self.__coef[2],self.__coef[3])
        else:
            print("optimize method is not yet run")


def demand_df(): # returns Date - Demand model dataframe
    conso = import_csv()
    L={}
    C={}
    N=conso.shape
    for j in range(N[0]):
        LDZ=conso.loc[j,'LDZ']
        E=conso.loc[j,'Date']
        C[E]=LDZ
    dff=pd.DataFrame(list(C.items()),columns=['Date', 'real_demand'])
    pickle.dump(dff, open("real_d.sav", 'wb'))

    for i in range(N[0]):
        K=conso.loc[i,'Date']
        A=conso.loc[i,'Actual']
        D=h(A,g[0],g[1],g[2],g[3])
        L[K]=D

    df=pd.DataFrame(list(L.items()),columns=['Date', 'Demand'])
    
    #dg=pd.DataFrame(g,columns=['Coeff'])
    #dg.to_csv("coef.csv",sep=";",index=False)
    
    return df

    #This is what the class print if you use the print fun500, -25, 2, 100
#If you have filled correctly the following code will run without an issue        
if __name__ == '__main__':

    #set working directory
    #set_wd()

    #1) import consumption data and plot it
    conso = import_csv()
    #g = [500, -25, 2, 100]
    #2) work on consumption data (non-linear regression)
    #2)1. Plot consumption as a function of temperature    

    scatter_plot()        

    #2)2. optimize the parameters
    sig = optimize_sigmoid(conso)
    g = sig.optimize()
    c = sig.create_consumption()
    ##sig = optimize_sigmoid(conso)
    #sig.optimize()
    #c = sig.create_consumption()
    #print(sig)





    #2)3. check the new fit

    # These are the 3 ways to access a protected attribute, it works the same for a protected method
    # An attribute/method is protected when it starts with 2 underscores "__"
    # Protection is good to not falsy create change
print(
            [
            sig.__dict__['_optimize_sigmoid__corr'],
            sig.__dict__['_optimize_sigmoid__rmse'],
            sig.__dict__['_optimize_sigmoid__nrmse'],
            sig.__dict__['_optimize_sigmoid__anrmse']
            ]
        )
print(
            [
            sig._optimize_sigmoid__corr,
            sig._optimize_sigmoid__rmse,
            sig._optimize_sigmoid__nrmse,
            sig._optimize_sigmoid__anrmse
            ]
        )

print(
            [
            getattr(sig, "_optimize_sigmoid__corr"),
            getattr(sig, "_optimize_sigmoid__rmse"),
            getattr(sig, "_optimize_sigmoid__nrmse"),
            getattr(sig, "_optimize_sigmoid__anrmse")
            ]
        )
    
print(sig.fit_metrics())
c.sigmoid(True)
print(c)

D_df = demand_df()
pickle.dump(D_df, open('Demand_model.sav', 'wb'))
    
    #3) If time allows do TSA on actual temperature
    #3)1. Check trend (and Remove it)
    #3)2. Check Seasonality (Normal Temperature)
    #3)3. Model stochastic part that is left with ARIMA
    #3)4. Use this to forecast consumption over N days

