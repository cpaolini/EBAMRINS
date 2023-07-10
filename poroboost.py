#
# C. Paolini
# paolini@engineering.sdsu.edu
# LBNL 06/14/23
#
# X_train: (2445, 2097152)
# X_test:  (815, 2097152)
# y_train: (2445, 5)
# y_test:  (815, 5)
#
import sys
import os
import math
import pickle 
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, target=None, batch_size=5):
        self.it = 0 
        self.df = df
        self.target = target
        self.batch_size = batch_size
        self.rows, self.cols = df.shape
        self.batches = int( np.ceil( self.rows / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        # If you need to perform iteration again from the beginning, 
        # you can initialize it with the reset() function.
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        # self.batches defined at class instance creation. It contains the total number of batches that can be passed.
        # End the iteration when self.it has reached the number of deliverable batches.
        if self.it == self.batches:
            # 
            return 0 
        
        # Get the start-end point for indexing.
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, self.rows )
        # Contain the data that will be passed by batch to input.
        input_data(data=self.df[a:b], label=self.target[a:b,2]) # component 0 
        self.it += 1
        return 1
dir='/global/cfs/projectdirs/m1516/summer2023'        
pklfile='/global/cfs/projectdirs/m1516/summer2023/porosity.pkl'
print(f"Opening and reading {pklfile}")
file = open(pklfile,'rb')
X_train = pickle.load(file)
X_test = pickle.load(file)
y_train = pickle.load(file)
y_test = pickle.load(file)
print(f"Done reading {pklfile}")
print(f"Instantiating xgb.QuantileDMatrix")
Xy_train = IterLoadForDMatrix(X_train, y_train)
dtrain = xgb.QuantileDMatrix(Xy_train, max_bin=256)


# Create regression matrices
#                                 component 1 is y_train[2]
#dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=False)
print(f"Instantiating xgb.DMatrix")
dtest_reg = xgb.DMatrix(X_test, y_test[:,2], enable_categorical=False)    
#dtrain_reg = xgb.QuantileDMatrix(X_train, y_train, enable_categorical=False)
#dtest_reg = xgb.QuantileDMatrix(X_test, y_test, enable_categorical=False)    

# Define hyperparameters
params = {
    "objective": "reg:squarederror",
    "tree_method": "gpu_hist",
    'subsample': 0.2,
    'sampling_method': 'gradient_based',
}

evals = [(dtrain, "train"), (dtest_reg, "validation")]
evals_result = {}
n = 100

print(f"Invoking xgb.train")
model = xgb.train(
    params=params,
    #dtrain=dtrain_reg,
    dtrain=dtrain,
    num_boost_round=n,
    evals=evals,
    evals_result=evals_result,
    verbose_eval=True  # Every rounds
)
print(f"End xgb.train")
# TODO: dtest_reg should only use y[2]

preds = model.predict(dtest_reg)
rmse = mean_squared_error(y_test[:,2], preds, squared=False)

print(f"RMSE of the base model: {rmse:.3f}")

epochs = len(evals_result['validation']['rmse'])
x_axis = range(0, epochs)

fig = plt.figure(figsize=(6,3))
fig.suptitle(f"XGBoost RMSE", fontsize=16)
ax = fig.add_subplot(111)
ax.set_title(f"RMSE vs Epoch")
ax.set_xlabel("Epoch",labelpad=10) 
ax.set_ylabel("RMSE",labelpad=10)

ax.plot(x_axis, evals_result['validation']['rmse'], label='Train')
ax.plot(x_axis, evals_result['validation']['rmse'], label='Test')

ax.legend()
print(f"Saving plot")
plt.savefig(dir + '/' + "poroboost" + '.png', bbox_inches='tight')
plt.close(fig)
print(f"Done")
