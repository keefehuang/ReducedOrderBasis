import numpy as np
import fbpca
import numpy as np 
from scipy.optimize import lsq_linear

def powern(n):
	num = 2
	while num < n:
		yield num
		num = int(num * 1.5)

def singleReducedOrderApproximation(A, V, snapshot, node_selection, nodes=None, isInput=False):
	if isInput:
		x_tilde = nodes
	else:
		x_tilde = 	A[  node_selection	, snapshot 	]

	x_real 	= 	A[  	  : 		, snapshot 	]
	V_red 	= 	V[  node_selection 	,     :		]
	
	x_red, res, rank, s = np.linalg.lstsq(V_red, x_tilde, rcond=None)
	x_r = V.dot(x_red.T)

	return x_r

# Performs a for loop over all snapshots in Binout data "A" and calls singleReducedOrderApproximation which runs the
# least squares approximation for a single snapshot
def reducedOrderApproximation(A, V, snapshot_selection, node_selection, isError=True, nodes=None, isInput=False):
	case_error = []
	error_r = np.array([])
	A_r = np.array([])
	for snapshot in snapshot_selection:
		if isInput:
			nodeSnapshot = nodes[:, snapshot]
		else:
			nodeSnapshot = None
		x_r = singleReducedOrderApproximation(A, V, snapshot, node_selection, nodes=nodeSnapshot, isInput=isInput)
		if isError:
			error_x = np.array(abs(x_r - A[: , snapshot]))
			
		case_error.append( np.linalg.norm(x_r - A[: , snapshot], 2) )
		if snapshot == 0:
			A_r = x_r
			if isError:
				error_r = error_x
		else:
			A_r = np.column_stack((A_r,x_r))
			if isError:
				error_r = np.column_stack((error_r,error_x))
	if isError:
		return error_r, case_error, A_r

	return case_error, A_r

# Performs the SVD on the Binout data "A" and calls reducedOrderApproximation to perform the Least Squares approximation for the
# reconstructed matrix.
def matrixReconstruction(A, snapshot_selection, node_selection, basisRequired=True, V=None, reducedOrderMethod='rSVD', reducedOrderParams=None, isError=True, nodes=None, isInput=False):
	A = A[0]
	nodes = nodes[0]
	if not V:
		if reducedOrderMethod == 'rSVD':
			if reducedOrderParams:
				k = reducedOrderParams[0] # sampling for rSVD
				n = reducedOrderParams[1] # power iterations
			else:
				k = 5
				n = 4
			# rsvd, only reduced order basis (ROB) V is used for this method
			(V, s , Vt) = fbpca.pca(A, k=k , n_iter=n)
			
		elif reducedOrderMethod == 'SVD':
			k = 120
			# svd, only reduced order basis (ROB) V is used for this method
			U, s , Vt = np.linalg.svd(A)
			V = U[:,:k]
			
		else:
			print("ERROR: No valid reduced order method provided!")
			return
	
	error_r, case_error, A_r = reducedOrderApproximation(A, V, snapshot_selection, node_selection, isError=isError, nodes=nodes, isInput=isInput)
	error = ([k, case_error])
	print("The mean error for the reconstructed A matrix with {} basis  is {}".format(k, np.mean(case_error)))

	if isError:
		return error_r, A_r

	return A_r


##############################################################################################

def matrixReconstructionWithVelocity(A, Vel, snapshot_selection, node_selection, basisRequired=True, V=None, reducedOrderMethod='rSVD', reducedOrderParams=None, isError=True, nodes=None, isInput=False, dt = 0.1):
	if not V:
		if reducedOrderMethod == 'rSVD':
			if reducedOrderParams:
				p = reducedOrderParams[0] # sampling for rSVD
				q = reducedOrderParams[1] # power iterations
				k = reducedOrderParams[2] # k chosen from experimentation. 1e-5 error reached with 63 bases and ALL nodes
			else:
				p = 15 # sampling for rSVD
				q = 10  # power iterations
				k = 10 # k chosen from experimentation. 1e-5 error reached with 63 bases and ALL nodes
			# rsvd, only reduced order basis (ROB) V is used for this method
			(U, s , Vt) = fbpca.pca( A, k=k , n_iter=q)
			V = U
			
		elif reducedOrderMethod == 'SVD':
			k = 120
			# svd, only reduced order basis (ROB) V is used for this method
			U, s , Vt = np.linalg.svd(A)
			V = U[:,:k]
			
		else:
			print("ERROR: No valid reduced order method provided!")
			return
	
	error_r, case_error, A_r = reducedOrderApproximationWithVelocity(A, V, Vel, snapshot_selection, node_selection, isError=isError, nodes=nodes, isInput=isInput)
	error = ([k, case_error])
	print("The mean error for the reconstructed A matrix with {} basis  is {}".format(k, np.mean(case_error)))

	if isError:
		return error_r, A_r

	return A_r

def reducedOrderApproximationWithVelocity(A, V, Vel, snapshot_selection, node_selection, isError=True, nodes=None, isInput=False, dt = 0.1):
	case_error = []
	error_r = np.array([])
	A_r = np.array([])
	for snapshot in snapshot_selection:
		if isInput:
			nodeSnapshot = nodes[:, snapshot]
		else:
			nodeSnapshot = None
		x_r = singleReducedOrderApproximationWithVelocity(A, V, Vel, snapshot, node_selection, nodes=nodeSnapshot, isInput=isInput)
		if isError:
			error_x = np.array(abs(x_r - A[: , snapshot]))
			
		case_error.append( np.linalg.norm(x_r - A[: , snapshot], 2) )
		if snapshot == 0:
			A_r = x_r
			if isError:
				error_r = error_x
		else:
			A_r = np.column_stack((A_r,x_r))
			if isError:
				error_r = np.column_stack((error_r,error_x))
	if isError:
		return error_r, case_error, A_r

	return case_error, A_r

def singleReducedOrderApproximationWithVelocity(A, V, Vel, snapshot, node_selection, nodes=None, isInput=False, dt=0.1):
	if isInput:
		x_tilde = nodes
	
	displacement_approximation = euler(A[: , snapshot-1], Vel[:, snapshot-1], dt)
	rel_error = 0.1
	lb = displacement_approximation*(1-rel_error)
	ub = displacement_approximation*(1+rel_error)
	# displacement_approximation = crank_nicol(A[: , snapshot-1], Vel[:, snapshot-1], Vel[:,snapshot+1], dt)

	x_real 	= 	A[  	  : 		, snapshot 	]
	V_red 	= 	V[  node_selection 	,     :		]

	x_red, cost, fun, optimality, active_mask, nit, status = lsq_linear(V_red, x_tilde, bounds=(lb, ub), lsmr_tol='auto', verbose=0)

	x_r = V.dot(x_red.T)

	return x_r

def euler(y, dydt, dt):
	return y + dydt*dt

def crank_nicol(y, dydt1, dydt2, dt):
	return y + (dydt1 + dydt2)/2*dt


	#########################################################################################################################

	# This funtion is intended to reconstruct the matrix using multiple variables
def multivariableMatrixReconstruction(Variables, snapshot_selection, node_selection, basisRequired=True, V=None, reducedOrderMethod='rSVD', reducedOrderParams=None, isError=True, nodes=None, isInput=False):
	num_variables = len(Variables)
	arrayLength = len(node_selection)

	A = Variables[0]
	nodes_selection = node_selection
	for num in range(1,num_variables):
		A = np.concatenate((A, Variables[num]))	
		nodes_selection = np.concatenate((nodes_selection, np.add(node_selection, Variables[0].shape[0])),axis=0)

	if not V:
		if reducedOrderMethod == 'rSVD':
			if reducedOrderParams:
				p = reducedOrderParams[0] # sampling for rSVD
				q = reducedOrderParams[1] # power iterations
				k = reducedOrderParams[2] # k chosen from experimentation. 1e-5 error reached with 63 bases and ALL nodes
			else:
				p = 15 # sampling for rSVD
				q = 4  # power iterations
				k = 15 # k chosen from experimentation. 1e-5 error reached with 63 bases and ALL nodes
			# rsvd, only reduced order basis (ROB) V is used for this method
			
			(U, s , Vt) = fbpca.pca( A , k=k , n_iter=4)
			V = U
			
		elif reducedOrderMethod == 'SVD':
			k = 10
			# svd, only reduced order basis (ROB) V is used for this method
			U, s , Vt = np.linalg.svd(A)
			V = U[:,:k]
			
		else:
			print("ERROR: No valid reduced order method provided!")
			return

	print('SVD calculation Completed')
	
	error_r, case_error, A_r = reducedOrderApproximation(A, V, snapshot_selection, nodes_selection, isError=isError, nodes=nodes, isInput=isInput)
	error = ([k, case_error])
	Disp_r = A_r[:Variables[0].shape[0],:]
	error_r = error_r[:Variables[0].shape[0],:]
	print("The mean error for the reconstructed A matrix with {} basis  is {}".format(k, np.mean(case_error)))

	if isError:
		return error_r, Disp_r

	return Disp_r