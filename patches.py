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

    pFile = re.compile(r'plot.nx128.step(\d+).2d.hdf5')
    pLevel = re.compile(r'level_(\d+)')
                
    a = np.empty([0,7])
    for file in listdir(path):
        #print(file)
        mFile = pFile.match(file)

        if mFile:
            stepNum = int(mFile.group(1))
            print(f'file: {file}, step: {stepNum}')
            hf_in = h5py.File(file, "r")
            root = hf_in["/"]

            dxArray = np.empty((2,))

            isPrinted = False
            for key in root.keys():
                #print(f'\nkey: {key}', end=' ')
                mLevel = pLevel.match(key)
                
                if mLevel:
                    levelNum = int(mLevel.group(1))
                    #print(f'level number: {levelNum}', end=' ')              # printing out each level number 
                    levelHandle = hf_in[mLevel.group(0) + "/"]           # Making title 
                    time = levelHandle.attrs["time"]
                    level = hf_in[mLevel.group(0)]                
                    dx = level.attrs["dx"]                          # Printing dx value 
                    dxArray[levelNum] = dx
                    #print(f'dx: {dx}', end=' ')
                    boxes = hf_in[mLevel.group(0) + "/boxes"]
                    data_attributes = hf_in[mLevel.group(0) + "/data_attributes"]
                    numBoxes = boxes.shape[0]
                    #print(f'number of boxes: {numBoxes}', end=' ')

                    # get velocity data here
                    prob_domain = level.attrs["prob_domain"]
                    X, Y = np.mgrid[prob_domain[1]:prob_domain[3]+1, prob_domain[0]:prob_domain[2]+1]
                    velocity0 = np.zeros(X.shape)

                    i = 0
                    for box in boxes:

                        #if levelNum > 0:
                        #    box = (box[0], box[1], box[2], box[3])                     
                        
                        # each box is a 4-tuple
                        X = (box[0] * dx, box[1] * dx), (box[2] - box[0] + 1) * dx
                        Y = (box[3] - box[1] + 1) * dx

                        #boxDim = (box[2] - box[0] + 1, box[3] - box[1] + 1)
                        # print(f'box {i}: {boxDim}, box : {box}', end=' ')
                
                        width = box[0] * dx, box[1] * dx, (box[2] - box[0] + 1) * dx
                        height =  (box[3] - box[1] + 1) * dx
                        boxDim = (X, Y)

                        v = velocity0[box[1]:box[3]+1,box[0]:box[2]+1]
                        stdev = np.std(v)

                        #print(f'box {boxDim}, origin box dim : {box}', end=' ')
                        l = [stepNum, levelNum, box[0] * dx, box[1] * dx,  (box[2] - box[0] + 1) * dx, (box[3] - box[1] + 1) * dx, stdev]
                        
                        print(l)

                        a = np.append(a, [l], axis=0)
                        if isPrinted == False:
                            isPrinted = True
                            print(l)
                            #print(a[-1,:])
                        
                        i +=1

            #print('\n')
    #print(a)
    #print(a.shape)
    file = 'patches.pkl'
    with open(file, 'wb') as f:
        pickle.dump(a, f)
    if os.path.exists(file):
        print (f'pickle file: {file}')
        file_stats = os.stat(file)
        print(f'File Size in Bytes is {file_stats.st_size}')
    else:
        print (f'failed to generate pickle file: {file}')