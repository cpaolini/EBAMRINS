import sys
import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.tri as mtri
from tabulate import tabulate
from matplotlib import cm
from pathlib import Path
import seaborn as sns
from math import exp
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

#(conda_plot2d) paolini@perlmutter:login05:~/EBAMRINS> python3.9 porosvm.py 2 /global/cfs/projectdirs/m1516/summer2023/30um_testcases/30um.pkl /global/cfs/projectdirs/m1516/summer2023/u30_u100_2zone/u30_u100_2zone.pkl /global/cfs/projectdirs/m1516/summer2023/u30_u100/u30_u100.pkl

#import os
#os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
import timeit
#import mkl
#mkl.set_num_threads(2)
start = timeit.default_timer()

if len(sys.argv) != 5:
    print(f'usage: poroboost.py component pklfile1 pklfile2 pklfile3\n')
    exit(0)
    
component = int(sys.argv[1])
dir='/global/cfs/projectdirs/m1516/summer2023'        
pklfile1 = sys.argv[2]
pklfile2 = sys.argv[3]
pklfile3 = sys.argv[4]

path = Path(pklfile1)
if path.is_file():
    print(f"Opening and reading {pklfile1}")
    file = open(pklfile1,'rb')
    X_train_1 = pickle.load(file)
    X_test_1 = pickle.load(file)
    y_train_1 = pickle.load(file)
    y_test_1 = pickle.load(file)
    print(f"Done reading {pklfile1}")
    print(f"X_train: {X_train_1.shape}, X_test: {X_test_1.shape}, y_train: {y_train_1.shape}, y_test: {y_test_1.shape}")
else:
    print(f'error: {pklfile1} not found\n')
    exit(0)

path = Path(pklfile2)
if path.is_file():
    print(f"Opening and reading {pklfile2}")
    file = open(pklfile2,'rb')
    X_train_2 = pickle.load(file)
    X_test_2 = pickle.load(file)
    y_train_2 = pickle.load(file)
    y_test_2 = pickle.load(file)
    print(f"Done reading {pklfile2}")
    print(f"X_train: {X_train_2.shape}, X_test: {X_test_2.shape}, y_train: {y_train_2.shape}, y_test: {y_test_2.shape}")
else:
    print(f'error: {pklfile2} not found\n')
    exit(0)

path = Path(pklfile3)
if path.is_file():
    print(f"Opening and reading {pklfile3}")
    file = open(pklfile3,'rb')
    X_train_3 = pickle.load(file)
    X_test_3 = pickle.load(file)
    y_train_3 = pickle.load(file)
    y_test_3 = pickle.load(file)
    print(f"Done reading {pklfile3}")
    print(f"X_train: {X_train_3.shape}, X_test: {X_test_3.shape}, y_train: {y_train_3.shape}, y_test: {y_test_3.shape}")
else:
    print(f'error: {pklfile3} not found\n')
    exit(0)

X_train = np.vstack((X_train_1, X_train_2, X_train_3))
X_test = np.vstack((X_test_1, X_test_2, X_test_3))
y_train = np.concatenate((y_train_1, y_train_2, y_train_3))
y_test = np.concatenate((y_test_1, y_test_2, y_test_3))
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")


#df = pd.read_csv("Soil_Airtemp.csv")


#dataset = df[['Volumetric Water Content','Electrical Conductivity (dS/m)','T (degrees celcius)','Depth (cm)','AirT']]
#print("Dataset Description",dataset.describe())

#x= dataset[['Volumetric Water Content','Electrical Conductivity (dS/m)','Depth (cm)','T (degrees celcius)']]
#y= dataset['AirT']



# Dividing the training and testing dataset into 70% in training and 30% testing
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25x0.8=0.2


#print("Dataset size    :",dataset.shape[0])
#print("Training   size :",X_train.shape[0])
#print("Validation size :",X_val.shape[0])
#print("Testing    size :",X_test.shape[0])


#from sklearn.svm import SVR
#from sklearn.model_selection import RandomizedSearchCV

#SVM

# Tuning of parameters for regression by cross-validation
#K =10 # Number of cross validation
'''
param = {'kernel' : ['rbf'],
         'C' : [0.1,1,10,100,1000],
         #'degree' : [1,2,3],
         #'coef0' : [0.00001,0.001,0.01,0.1,10,0.5],
         'gamma' : [1e-7, 1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000,100000],
         'epsilon': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]},


param = {'kernel' : ['rbf'],
         'C' : [0.1,1,10,100,1000],
         #'degree' : [1,2,3],
         #'coef0' : [0.00001,0.001,0.01,0.1,10,0.5],
         'gamma' : [1e-7, 1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000,100000],
        'epsilon': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
       },

param = {'kernel' : ['rbf'],
         'C' : [0.1,1,10,100,1000],
         #'degree' : [1,2,3],
         #'coef0' : [0.00001,0.001,0.01,0.1,10,0.5],
         'gamma' : [1e-2,1e-1,1,10,100,1000,10000],
        'epsilon': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
       },
param = {'kernel' : ['rbf'],
         'C' : [10,50,100,150,200,250,300,350,400,450,500],
         #'degree' : [1,2,3],
         #'coef0' : [0.00001,0.001,0.01,0.1,10,0.5],
         'gamma' : [1e-2,1e-1,1,10,100,1000,10000],
        'epsilon': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
       },
param = {'kernel' : ['rbf'],
         'C' : [25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500],
         #'degree' : [1,2,3],
         #'coef0' : [0.00001,0.001,0.01,0.1,10,0.5],
         'gamma' : [1e-2,1e-1,1,10,100,1000,10000],
        'epsilon': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
       },
param = {'kernel' : ['rbf'],
         'C' : [5,10,15,20,25,50,75,100,125,150,175,200,225,250],
         #'degree' : [1,2,3],
         #'coef0' : [0.00001,0.001,0.01,0.1,10,0.5],
         'gamma' : [1e-2,1e-1,1,10,50,75,100,125,150,250,500,1000],
        'epsilon': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
       },
param = {'kernel' : ['rbf'],
         'C' : [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
         #'degree' : [1,2,3],
         #'coef0' : [0.00001,0.001,0.01,0.1,10,0.5],
         'gamma' : [1e-2,1e-1,1,10,50,75,100,125,150,250,500,1000],
        'epsilon': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
       },
param = {'kernel' : ['rbf'],
         'C' : [50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135],
         'gamma' : [1e-2,1e-1,1,10,50,75,100,125,150,250,500,1000],
        'epsilon': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
       },
param = {'kernel' : ['rbf'],
         'C' : [50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135],
         'gamma' : [1e-2,1e-1,1,10,50,75,100,125,150,250,500,1000],
        'epsilon': [0.5,0.6,0.7,0.8,0.9]
       },
param = {'kernel' : ['rbf']
         }
svr = SVR()
'''

regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

#svr_model = GridSearchCV(svr, param, cv = K, n_jobs = -1, verbose = 0)
#svr_model = RandomizedSearchCV(svr,param,n_iter=10,cv = [(slice(None), slice(None))],verbose=1)
#svr_model = RandomizedSearchCV(svr,param,n_iter=250,cv = 5,verbose=1)
#svr_model = RandomizedSearchCV(svr,param,n_iter=2000,cv = 5,verbose=1)
#svr_model.fit(X_train, y_train)

#print("Hyperparameters of SVM Regressor",param)
#print("Best Param",svr_model.best_params_)
#print("***********************************")
pred_y_test= regressor.predict(X_test)

# Testing set
r31 = mean_squared_error(y_test,pred_y_test)
r32 = sqrt(mean_squared_error(y_test,pred_y_test))
r33 = mean_absolute_error(y_test,pred_y_test) 
    
data=[["Testing Set  ",r31,r32,r33]]
columns=["Dataset","MSE","RMSE","MAE"]
print(tabulate(data,headers=columns, tablefmt="fancy_grid"))

stop = timeit.default_timer()
print("Run Time(seconds): ", stop-start)  


#filename ="best_svm_model.pickle"
#pickle.dump(svr_model.best_estimator_,open(filename,'wb'))

#loadmodel=pickle.load(open(filename,'rb'))
#predictedvalue=loadmodel.predict(X_test)
#print("MAE:",mean_absolute_error(y_test,predictedvalue))
