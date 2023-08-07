#
# C. Paolini
# paolini@engineering.sdsu.edu
# LBNL 06/14/23
#
import sys
import os
import math
import pickle 
import numpy as np
import matplotlib.pyplot as plt

pklfile='/global/cfs/projectdirs/m1516/summer2023/porosity.pkl'
file = open(pklfile,'rb')
X_train = pickle.load(file)
X_test = pickle.load(file)
y_train = pickle.load(file)
y_test = pickle.load(file)

rows = X_train.shape[0]
for row in range(0, rows):
    a = np.reshape(X_train[row], (1024, 2048))
    fig = plt.figure(figsize=(6,3))
    fig.suptitle(f"minDistSph: {y_train[row][0]}", fontsize=16)
    ax = fig.add_subplot(111)
    ax.pcolor(a,cmap='hot')
    ax.set_title(f"porosity0")
    ax.set_xlabel("x",labelpad=10) 
    ax.set_ylabel("y",labelpad=10)
    plt.savefig("train_" + str(y_train[row][0]) + '_' + str(row) + '.png', bbox_inches='tight')
    plt.close(fig)
    print("train_" + str(y_train[row][0]) + '_' + str(row))
    

