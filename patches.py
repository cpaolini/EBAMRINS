import sys
import os
import h5py
import math
import re
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pickle
from os import listdir
from os.path import isfile, join
import glob

def sorted_directory(directory):
    items = glob.glob(directory + 'plot.*')
    sorted_items = sorted(items)
    return sorted_items

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print (f'usage: patches.py <path>')
        sys.exit()
    else:
        path = sys.argv[1]
        if os.path.isdir(path) == False:
            print (f'{path} is not a valid directory')
            sys.exit()
        else:
            os.chdir(path)
            
pFile = re.compile('plot.nx128.step(\d+).2d.hdf5')
pLevel = re.compile('level_(\d+)')
a = np.empty([0,8])


i = 0

for file in sorted_directory(path):
#for file in listdir(path):
    mFile = pFile.match(file)

    if mFile:
        stepNum = int(mFile.group(1))
        print(f'file: {file}, step: {stepNum}')
        hf_in = h5py.File(file, "r")
        root = hf_in["/"]

        attributes = hf_in["level_0/data_attributes/"]
        data = hf_in["level_0/data:datatype=0"]
        componentKeys = list(root.attrs.keys())
        comps = attributes.attrs["comps"]
        components = [i.decode('utf-8') if isinstance(i, np.bytes_) else '' for i in list(root.attrs.values())]
        dataNumPy = np.array(data, np.float64)
        
        level_0 = hf_in["level_0/"]
        prob_domain = level_0.attrs["prob_domain"]
        X, Y = np.mgrid[prob_domain[1]:prob_domain[3]+1, prob_domain[0]:prob_domain[2]+1]
                
        # Box dimensions
        boxes = hf_in["level_0/boxes"]
        boxDim = (boxes[0][2] - boxes[0][0] + 1, boxes[0][3] - boxes[0][1] + 1) # assume level 0 boxes are all of equivalent dimension
        nBoxes = boxes.shape[0]
        nRows = X.shape[0]
        nCols = X.shape[1]
        patchRows = int(nRows/boxDim[1])
        patchCols = int(nCols/boxDim[0])
        boxData = dataNumPy.reshape((nBoxes,int(dataNumPy.shape[0]/nBoxes)))
                
        # Velocity extraction
        velocity0_i = int(re.findall(r'\d+', componentKeys[components.index("velocity0")])[0]) 

        velocity0 = np.zeros(X.shape)
        #print(f"velocity0.shape {velocity0.shape}")
                
        #print(f"patchCols {patchCols}")
        for col in range(patchCols):
            #print(f"patchRows {patchRows}")
            for row in range(patchRows):
                #   print(f"[{row * boxDim[0]} : {(row + 1) * boxDim[0]}, {col * boxDim[1]} : {(col + 1) * boxDim[1]}]")
                velocity0[row * boxDim[0] : (row + 1) * boxDim[0], col * boxDim[1] : (col + 1) * boxDim[1]] = \
                    np.lib.stride_tricks.as_strided(boxData[col * patchRows + row, velocity0_i*(boxDim[0] * boxDim[1]):(velocity0_i+1)*(boxDim[0] * boxDim[1])], \
                    shape=(boxDim[0],boxDim[1]),\
                    strides=(8 * boxDim[1], 8 * 1))   
        
        i += 1
        dxArray = np.empty((0,))
        for key in root.keys():
            #print(f'\nkey: {key}', end=' ')
            mLevel = pLevel.match(key)
            
            if mLevel:
                levelNum = int(mLevel.group(1))

                #print(f'level number: {levelNum}', end=' ')         # printing out each level number 
                levelHandle = hf_in[mLevel.group(0) + "/"]           # Making title 
                time = levelHandle.attrs["time"]
                level = hf_in[mLevel.group(0)]                
                
                dx = level.attrs["dx"]                               # Printing dx value 
                dxArray = np.append(dxArray,dx)
                #print(f'dx: {dx}')
                
                boxes = hf_in[mLevel.group(0) + "/boxes"]
                data_attributes = hf_in[mLevel.group(0) + "/data_attributes"]
                numBoxes = boxes.shape[0]
                #print(f'number of boxes: {numBoxes}', end=' ')   


                i = 0                       
                for box in boxes:
    
                    boxDim = (box[2] - box[0] + 1, box[3] - box[1] + 1)                
                    
                    # print(f'box {i}: {boxDim}, box : {box}', end=' ')

                    width = box[0] * dx, box[1] * dx, (box[2] - box[0] + 1) * dx
                    height = (box[3] - box[1] + 1) * dx
                    
                    comps == int(data.shape[0]/((boxDim[0] * boxDim[1]) * nBoxes ))
                    dataNumPy.shape[0]/nBoxes == boxDim[0] * boxDim[1] * comps
                    boxData = dataNumPy.reshape((nBoxes,int(dataNumPy.shape[0]/nBoxes)))
                    factor = 2**levelNum
                    v = velocity0[box[1]//factor:(box[3]//factor)+1,box[0]//factor:(box[2]//factor)+1]
                    #print(f"v {v.shape}: {v}")
                    #print(np.max(v))
                    stdev = np.std(v) 

                    if math.isnan(stdev):
                        print(f'nan v : {v}, v dim: {v.shape}, {box[1]}:{box[3]+1},{box[0]}:{box[2]+1}')  
                        print(f"level: {levelNum}")
                        print(l)
                        print(f"v {v.shape}: {v}, velocity0: {velocity0.shape}\n\n\n")
                        sys.exit(1)

                    #print(f'box {boxDim}, origin box dim : {box}', end=' ')
                    l = [stepNum, levelNum, box[0] * dx, box[1] * dx,  (box[2] - box[0] + 1) * dx, (box[3] - box[1] + 1) * dx, stdev, i ]
                    #print(l)
                    a = np.append(a, [l], axis=0)       

                    i =+ 0            

with open('patches.pkl', 'wb') as f:
    pickle.dump(a, f)
