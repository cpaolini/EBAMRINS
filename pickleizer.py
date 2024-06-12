#
# C. Paolini
# paolini@engineering.sdsu.edu
# LBNL 06/14/23
#
#module load python
#conda create -n plot2d python=3.9 pip numpy h5py matplotlib
#conda create -n conda_plot2d python=3.9 pip numpy h5py matplotlib
#conda activate conda_plot2d
#conda deactivate conda_plot2d
#conda deactivate
#conda activate /global/cfs/projectdirs/m1516/summer2023/conda_plot2d
#ls */conda*
#conda create -n conda_plot2d python=3.9 pip numpy h5py matplotlib
#conda activate conda_plot2d
#conda deactivate
#conda activate conda_plot2d
#conda activate conda_plot2d
#conda deactivate
#conda -h
#conda --help
#conda list
#ls -la ~/.conda/
#conda list -n conda_plot2d
#conda install -n conda_plot2d -c conda-forge py-xgboost-gpu
#conda install -n conda_plot2d scikit-learn
# 
import sys
import os
import h5py
import math
import re
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import xgboost as xgb
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=DeprecationWarning) 

rx_dict = {
    'minDistSph': re.compile(r'minDistSph = (?P<minDistSph>[0-9]?\.[0-9]+)\n'),
    'minDistCyl': re.compile(r'^minDistCyl = (?P<minDistCyl>[0-9]?\.[0-9]+)\n'),
    'step': re.compile(r'EBAMRNoSubcycle: step (?P<step>\d+)\n'),
    'rate': re.compile(r'Average reaction rate mineral (?P<mineral>\d+) component (?P<component>\d+): (?P<rate>[+-]?[0-9]+\.[0-9]+[e]?[+-]?[0-9]+?)\n'),
}

def _parse_line(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None

#rootdir = '/global/cfs/projectdirs/m1516/summer2023'

if len(sys.argv) != 4:
    print(f'usage: pickleizer.py rootdir pklfile\n')
    exit(0)
else:
    rootdir = sys.argv[1]
    if os.path.exists(rootdir) == False:
        print(f'error: rootdir does not exist\n')
        exit(0)
    pklfile = sys.argv[2] #'/global/cfs/projectdirs/m1516/summer2023/porosity.pkl'
    inputs = sys.argv[3]
    
A = []
y = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # print(os.path.join(subdir, file))
        split_tup = os.path.splitext(os.path.join(subdir, file))
  
        # extract the file name and extension
        file_name = split_tup[0]
        file_extension = split_tup[1]
  
        if file_extension == '.hdf5' and '.step0004000.2d' in file_name:
            print("File Name: ", file_name)
            #print("File Extension: ", file_extension)
            #print(file_name[:-2])
            
            if os.path.isfile(os.path.join(subdir, file)):
                hf_in = h5py.File(os.path.join(subdir, file),"r")
                root = hf_in["/"]
                data = hf_in["level_0/data:datatype=0"]
                offsets = hf_in["level_0/data:offsets=0"]
                attributes = hf_in["level_0/data_attributes/"]
                boxes = hf_in["level_0/boxes"]
                nBoxes = boxes.shape[0]
                boxDim = (boxes[0][2] - boxes[0][0] + 1, boxes[0][3] - boxes[0][1] + 1)
                #print(f'box dimension: {boxDim}')
                comps = attributes.attrs["comps"]
                #print(f'components: {comps}')

                components = [i.decode('utf-8') if isinstance(i, np.bytes_) else '' for i in list(root.attrs.values())]

                level_0 = hf_in["level_0/"]
                prob_domain = level_0.attrs["prob_domain"]
                time = level_0.attrs["time"]
                dataNumPy = np.array(data, np.float64)
                boxData = dataNumPy.reshape((nBoxes,int(dataNumPy.shape[0]/nBoxes)))
                #print(f'data dimension: {boxData.shape}')

                X, Y = np.mgrid[prob_domain[1]:prob_domain[3]+1, prob_domain[0]:prob_domain[2]+1]
                nRows = X.shape[0]
                nCols = X.shape[1]

                patchRows = int(nRows/boxDim[1])
                patchCols = int(nCols/boxDim[0])

                #print(f'grid dimension: {X.shape}')
                #print(f'patch columns: {patchCols}')
                #print(f'patch rows: {patchRows}')

                #fig = plt.figure(figsize=(6,9))
                #fig.suptitle(f"{filename}, time = {time:.4f}s", fontsize=16)

                velocity0_i = components.index("velocity0")
                velocity1_i = components.index("velocity1")
                porosity0_i = components.index("porosity0")

                componentKeys = list(root.attrs.keys())
                componentValues = list(root.attrs.values())
                velocity0_i = int(re.findall(r'\d+', componentKeys[velocity0_i])[0]) 
                velocity1_i = int(re.findall(r'\d+', componentKeys[velocity1_i])[0]) 
                porosity0_i = int(re.findall(r'\d+', componentKeys[porosity0_i])[0]) 

                velocity0 = np.zeros(X.shape)
                for col in range(patchCols):
                    for row in range(patchRows):
                        velocity0[row * boxDim[0] : (row + 1) * boxDim[0], col * boxDim[1] : (col + 1) * boxDim[1]] = \
                            np.lib.stride_tricks.as_strided(boxData[col * patchRows + row], shape=(boxDim[0],boxDim[1]), strides=(8*boxDim[1],8*1))
                #ax1 = fig.add_subplot(311)
                #ax1.pcolor(velocity0,cmap='hot')
                #ax1.set_title(f"velocity0")
                #ax1.set_xlabel("x",labelpad=10)
                #ax1.set_ylabel("y",labelpad=10)

                velocity1 = np.zeros(X.shape)
                for col in range(patchCols):
                    for row in range(patchRows):
                        velocity1[row * boxDim[0] : (row + 1) * boxDim[0], col * boxDim[1] : (col + 1) * boxDim[1]] = \
                            np.lib.stride_tricks.as_strided(boxData[col * patchRows + row, 1*(boxDim[0] * boxDim[1]):2*(boxDim[0] * boxDim[1])], shape=(boxDim[0],boxDim[1]), strides=(8*boxDim[1],8*1))
                #ax2 = fig.add_subplot(312)
                #ax2.pcolor(velocity1,cmap='hot')
                #ax2.set_title(f"velocity1")
                #ax2.set_xlabel("x",labelpad=10) 
                #ax2.set_ylabel("y",labelpad=10)

                porosity0 = np.zeros(X.shape)
                for col in range(patchCols):
                    for row in range(patchRows):
                        porosity0[row * boxDim[0] : (row + 1) * boxDim[0], col * boxDim[1] : (col + 1) * boxDim[1]] = \
                            np.lib.stride_tricks.as_strided(boxData[col * patchRows + row, porosity0_i*(boxDim[0] * boxDim[1]):(porosity0_i+1)*(boxDim[0] * boxDim[1])], shape=(boxDim[0],boxDim[1]), strides=(8*boxDim[1],8*1))
                #ax3 = fig.add_subplot(313)
                #ax3.pcolor(porosity0,cmap='hot')
                #ax3.set_title(f"porosity0")
                #ax3.set_xlabel("x",labelpad=10) 
                #ax3.set_ylabel("y",labelpad=10)

                #plt.show()
                #plt.savefig(filename + '.png', bbox_inches='tight')
                #plt.close(fig)

                ###B = np.dstack((velocity0,velocity1))
                ###B = np.dstack((B,porosity0))

                B = np.copy(porosity0)
                print(f'B.shape: {B.shape}')
                A.append(B.reshape(-1))
                print(f'len(A): {len(A)}')
                # packedChannel001.inputs_01
                b = np.zeros((5,))
                with open(os.path.join(subdir, inputs), 'r') as file_object:
                    line = file_object.readline()
                    while line:
                        key, match = _parse_line(line)
                        #print(f'key={key}')

                        if key == 'minDistSph':
                            minDistSph = match.group('minDistSph')
                            print(f'minDistSph: {minDistSph}')
                            b[0] = float(minDistSph)

                        if key == 'minDistCyl':
                            minDistCyl = match.group('minDistCyl')
                            print(f'minDistCyl: {minDistCyl}')
                            b[1] = float(minDistCyl)

                        #print(line)
                        line = file_object.readline()
                        lastStep = False
                with open(os.path.join(subdir, 'pout.0'), 'r') as file_object:
                    line = file_object.readline()
                    while line:
                        key, match = _parse_line(line)
                        #print(f'key={key}')

                        if key == 'step':
                            step = match.group('step')
                            if step == '4000':
                                print(f'step: {step}')
                                lastStep = True
                            else:
                                lastStep = False

                        if key == 'rate' and lastStep:
                            mineral = match.group('mineral')
                            component = match.group('component')
                            rate = match.group('rate')
                            b[int(component)+2] = float(rate)

                            print(f'rate: {rate} {component} {mineral}')

                        #print(line)
                        line = file_object.readline()
                    y.append(b)
            else:
                raise Exception(f"file {os.path.join(subdir, file)} not found")
            
X = np.array(A)
y = np.array(y)
print(f'X.shape: {X.shape}')
print(f'y.shape: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(f'X_train.shape: {X_train.shape}')
print(f'X_test.shape: {X_test.shape}')
print(f'y_train.shape: {y_train.shape}')
print(f'y_test.shape: {y_test.shape}')

with open(pklfile, 'wb') as f:
    pickle.dump(X_train, f)
    pickle.dump(X_test, f)
    pickle.dump(y_train, f)
    pickle.dump(y_test, f)
    f.close()

print(pklfile)
file = open(pklfile,'rb')
X_train = pickle.load(file)
X_test = pickle.load(file)
y_train = pickle.load(file)
y_test = pickle.load(file)
print(f'X_train: {X_train.shape}')
print(f'X_test:  {X_test.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_test:  {y_test.shape}')
file.close()
hist,bins = np.histogram(y_train[:,1],bins=np.linspace(0.001,0.010,10)) 
print(f'minDistSph bins: {bins}')  
print(f'training frequency distribution: {hist}') 
hist,bins = np.histogram(y_test[:,1],bins=np.linspace(0.001,0.010,10)) 
print(f'testing frequency distribution:  {hist}') 
