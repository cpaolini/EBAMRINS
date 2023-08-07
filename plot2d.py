#
# C. Paolini
# paolini@engineering.sdsu.edu
# LBNL 06/14/23
#
import sys
import os
import h5py
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

# Get the filename from the command line
filename = sys.argv[1]

if os.path.isfile(filename):
	hf_in = h5py.File(filename,"r")
	root = hf_in["/"]
	data = hf_in["level_0/data:datatype=0"]
	offsets = hf_in["level_0/data:offsets=0"]
	attributes = hf_in["level_0/data_attributes/"]
	boxes = hf_in["level_0/boxes"]
	nBoxes = boxes.shape[0]
	boxDim = (boxes[0][2] - boxes[0][0] + 1, boxes[0][3] - boxes[0][1] + 1)
	print(f'box dimension: {boxDim}')
	comps = attributes.attrs["comps"]
	print(f'components: {comps}')

	components = [i.decode('utf-8') if isinstance(i, np.bytes_) else '' for i in list(root.attrs.values())]

	level_0 = hf_in["level_0/"]
	prob_domain = level_0.attrs["prob_domain"]
	time = level_0.attrs["time"]
	dataNumPy = np.array(data, np.float64)
	boxData = dataNumPy.reshape((nBoxes,int(dataNumPy.shape[0]/nBoxes)))
	print(f'data dimension: {boxData.shape}')

	X, Y = np.mgrid[prob_domain[1]:prob_domain[3]+1, prob_domain[0]:prob_domain[2]+1]
	nRows = X.shape[0]
	nCols = X.shape[1]

	patchRows = int(nRows/boxDim[1])
	patchCols = int(nCols/boxDim[0])

	print(f'grid dimension: {X.shape}')
	print(f'patch columns: {patchCols}')
	print(f'patch rows: {patchRows}')

	fig = plt.figure(figsize=(6,12))
	fig.suptitle(f"{filename}, time = {time:.4f}s", fontsize=16)

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
	ax1 = fig.add_subplot(311)
	ax1.pcolor(velocity0,cmap='hot')
	ax1.set_title(f"velocity0")
	ax1.set_xlabel("x",labelpad=10)
	ax1.set_ylabel("y",labelpad=10)

	velocity1 = np.zeros(X.shape)
	for col in range(patchCols):
		for row in range(patchRows):
			velocity1[row * boxDim[0] : (row + 1) * boxDim[0], col * boxDim[1] : (col + 1) * boxDim[1]] = \
				np.lib.stride_tricks.as_strided(boxData[col * patchRows + row, 1*(boxDim[0] * boxDim[1]):2*(boxDim[0] * boxDim[1])], shape=(boxDim[0],boxDim[1]), strides=(8*boxDim[1],8*1))
	ax2 = fig.add_subplot(312)
	ax2.pcolor(velocity1,cmap='hot')
	ax2.set_title(f"velocity1")
	ax2.set_xlabel("x",labelpad=10) 
	ax2.set_ylabel("y",labelpad=10)

	porosity0 = np.zeros(X.shape)
	for col in range(patchCols):
		for row in range(patchRows):
			porosity0[row * boxDim[0] : (row + 1) * boxDim[0], col * boxDim[1] : (col + 1) * boxDim[1]] = \
				np.lib.stride_tricks.as_strided(boxData[col * patchRows + row, porosity0_i*(boxDim[0] * boxDim[1]):(porosity0_i+1)*(boxDim[0] * boxDim[1])], shape=(boxDim[0],boxDim[1]), strides=(8*boxDim[1],8*1))
	ax3 = fig.add_subplot(313)
	ax3.pcolor(porosity0,cmap='hot')
	ax3.set_title(f"porosity0")
	ax3.set_xlabel("x",labelpad=10) 
	ax3.set_ylabel("y",labelpad=10)

	#plt.show()
	plt.savefig(filename + '.png', bbox_inches='tight', dpi=2400)
	plt.close(fig)

else:
	raise Exception(f"file {filename} not found")
