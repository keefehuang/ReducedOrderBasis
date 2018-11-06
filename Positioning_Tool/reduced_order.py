# Add files in the parent folder to the path
import sys
import os
from os.path import splitext
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import fbpca
import numpy as np 
from scipy.optimize import lsq_linear

def powern(n):
	num = 2
	while num < n:
		yield num
		num = int(num * 1.5)

def reducedOrderApproximation(V, node_selection, nodes=None):

	V_red 	= 	V[  node_selection 	,     :		]
	
	x_red, res, rank, s = np.linalg.lstsq(V_red, nodes)
	x_r = V.dot(x_red)

	return x_r