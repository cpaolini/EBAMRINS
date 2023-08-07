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
#  1.0 Ca++ - 2.0 H+ + 1.0 CO2(aq) + 1.0 H2O <-> 1.0 CaCO3
# Average reaction rate mineral 0 component 0: Calcite CaCO3 
#
#
import sys
import os
import math
import pickle 
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn.metrics import mean_squared_error

class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, target=None, batch_size=5):
        self.it = 0 
        self.df = df
        self.target = target
        self.batch_size = batch_size
        self.rows, self.cols = df.shape
        self.batches = int( np.ceil( self.rows / self.batch_size ) )
        print(f"batch_size = {batch_size}")
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
#        input_data(data=self.df[a:b], label=self.target[a:b,2]) # component 0

        data=self.df[a:b]
        label=self.target[a:b]
        #print(f"next batch rows [{a},{b}), data dimension {data.shape}, target dimension {label.shape}, iteration {self.it}")
        #print(f"labels: {label}, type of {label[0]}: {type(label[0])}")
        input_data(data=self.df[a:b], label=self.target[a:b])
        self.it += 1
        return 1

# u30_u100_2zone> python3.9 poroboost.py 2 ../u30_u100/u30_u100.pkl ./u30_u100_2zone.pkl ../30um_testcases/30um.pkl

    
if len(sys.argv) != 5:
    print(f'usage: poroboost.py component pklfile1 pklfile2 pklfile3\n')
    exit(0)
    
component = int(sys.argv[1])
dir='/global/cfs/projectdirs/m1516/summer2023'        
pklfile1 = sys.argv[2]
pklfile2 = sys.argv[3]
pklfile3 = sys.argv[4]

print(f"Opening and reading {pklfile1}")
file = open(pklfile1,'rb')
X_train_1 = pickle.load(file)
X_test_1 = pickle.load(file)
y_train_1 = pickle.load(file)
y_test_1 = pickle.load(file)
print(f"Done reading {pklfile1}")
print(f"X_train: {X_train_1.shape}, X_test: {X_test_1.shape}, y_train: {y_train_1.shape}, y_test: {y_test_1.shape}")

print(f"Opening and reading {pklfile2}")
file = open(pklfile2,'rb')
X_train_2 = pickle.load(file)
X_test_2 = pickle.load(file)
y_train_2 = pickle.load(file)
y_test_2 = pickle.load(file)
print(f"Done reading {pklfile2}")
print(f"X_train: {X_train_2.shape}, X_test: {X_test_2.shape}, y_train: {y_train_2.shape}, y_test: {y_test_2.shape}")

print(f"Opening and reading {pklfile3}")
file = open(pklfile3,'rb')
X_train_3 = pickle.load(file)
X_test_3 = pickle.load(file)
y_train_3 = pickle.load(file)
y_test_3 = pickle.load(file)
print(f"Done reading {pklfile3}")
print(f"X_train: {X_train_3.shape}, X_test: {X_test_3.shape}, y_train: {y_train_3.shape}, y_test: {y_test_3.shape}")

X_train = np.vstack((X_train_1, X_train_2, X_train_3))
X_test = np.vstack((X_test_1, X_test_2, X_test_3))
y_train = np.concatenate((y_train_1, y_train_2, y_train_3))
y_test = np.concatenate((y_test_1, y_test_2, y_test_3))
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

#print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
#print(f"Pickle types: X_train[0]: {type(X_train[0])}, y_train[0]: {type(y_train[0])}")
#print(f"Instantiating xgb.QuantileDMatrix")
#Xy_train = IterLoadForDMatrix(X_train, y_train, batch_size=20)
#dtrain = xgb.QuantileDMatrix(Xy_train, max_bin=256)

print(f"Instantiating xgb.DMatrix")
# Create regression matrices
#                                 component 1 is y_train[2]
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=False)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=False)
#dtest_reg = xgb.DMatrix(X_test, y_test[:,2+component], enable_categorical=False)    

#dtrain_reg = xgb.QuantileDMatrix(X_train, y_train, enable_categorical=False)
#dtest_reg = xgb.QuantileDMatrix(X_test, y_test, enable_categorical=False)    

# Define hyperparameters
params = {
    "objective": "reg:squarederror",
    "tree_method": "exact", # https://xgboost.readthedocs.io/en/stable/treemethod.html
#    "tree_method": "gpu_hist",
    'sampling_method': 'gradient_based',
}

#    'subsample': 0.2,

evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
#evals = [(dtrain, "train"), (dtest_reg, "validation")]
evals_result = {}
n = 20

print(f"Invoking xgb.train")
# returns a Booster (a trained booster model)
model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    #dtrain=dtrain,
    num_boost_round=n,
    evals=evals,
    evals_result=evals_result,
    verbose_eval=True  # Every rounds
)
print(f"End xgb.train")
# TODO: dtest_reg should only use y[2]

preds = model.predict(dtest_reg)
rmse = mean_squared_error(y_test, preds, squared=False)
#rmse = mean_squared_error(y_test[:,2+component], preds, squared=False)

print(f"RMSE of the base model: {rmse:.3f}")

epochs = len(evals_result['validation']['rmse'])
x_axis = range(0, epochs)

fig = plt.figure(figsize=(7.00, 3.50))

fig.suptitle(f"XGBoost: Average reaction rate for Calcite $(CaCO_{3})$ Component {component}", fontsize=12)
ax = fig.add_subplot(111)
ax.set_title(f"RMSE vs Epoch")
ax.set_xlabel("Epoch",labelpad=10) 
ax.set_ylabel("RMSE",labelpad=10)

ax.plot(x_axis, evals_result['train']['rmse'], label='Train', linestyle='dashed', linewidth=2.5)
ax.plot(x_axis, evals_result['validation']['rmse'], label='Test', linewidth=1.5)

ax.legend()
print(f"Saving plot")
plt.savefig("poroboost_" + str(component) + '.png', bbox_inches='tight', dpi=1200)
plt.close(fig)
print(f"Done")

fig, ax = plt.subplots(figsize=(30, 30))
xgb.plot_tree(model, num_trees=0, ax=ax, rankdir='LR')
plt.savefig("poroboost_tree_0_comp_" + str(component) + '.png', bbox_inches='tight', dpi=1200)

model.save_model("poroboost_component_" + str(component) + '_model.json')

dump_list = model.get_dump(with_stats=True, dump_format='text')
num_trees = len(dump_list)
print("dump_list")
print(dump_list)
