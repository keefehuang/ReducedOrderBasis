import numpy as np
import fbpca
import numpy as np 
from scipy.optimize import least_squares
import pyprind

time = 0.01
def powern(n):
	num = 2
	while num < n:
		yield num
		num = int(num * 1.5)

# This function solves a least squares optimization problem.
# The base problem is given as V_simplified * x_simplified = rhs_simplified
def least_squares_approximation(V, V_simplified, rhs_simplified):
	rhs_approx, res, rank, s = np.linalg.lstsq(V_simplified, rhs_simplified, rcond=None)
	rhs_reconstructed = V.dot(rhs_approx)

	return rhs_reconstructed

def cost_function(x, rhs_old, rhs_simplified, velocity_simplified, V, V_simplified, timestep):

	return V_simplified*x - x_simplified + (rhs_old - V_simplified*x)/timestep - velocity_simplified

# This function solves a least squares optimization problem with given cost function defined in "cost_function".
# The base problem is given as V_simplified * x_simplified = rhs_simplified
# The cost function attempts to penalize unphysical changes in velocity
def least_squares_approximation_with_velocity(V, V_simplified, rhs_simplified, velocity, rhs_old, timestep):
	if rhs_old is None:
		rhs_approx, res, rank, s = np.linalg.lstsq(V_simplified, rhs_simplified, rcond=None)
	else:
		rhs_approx, cost, res, jac, grad, opt, act_mask = least_squares(cost_function, np.zeros(V_simplified.shape[1]), args=(rhs_old, rhs_simplified, velocity, V, V_simplified, timestep))
	rhs_reconstructed = V.dot(rhs_approx)

	return rhs_reconstructed

# Performs a for loop over all snapshots in Binout data "A" and calls least_squares_approximation which runs the
# least squares approximation for a single snapshot
def reduced_order_approximation(V, snapshot_selection, node_selection, isCalculateError=False, nodes=None, isInput=False, A=None, velocity_data=None, timestep=None):
	error_reconstructed = None
	V_simplified = V[node_selection, :]
	if velocity_data is not None:
		# velocity_simplified = velocity_data[node_selection, :]
		reconstruction_title = "Reconstructing snapshots with velocity"
		x_old = None
	else:
		reconstruction_title = "Reconstructing snapshots"

	if isCalculateError and A is None:
		print("No A matrix provided for error calculation. No error will be output")
		isCalculateError = False
	bar = pyprind.ProgBar(len(snapshot_selection), monitor=True, title=reconstruction_title, bar_char='█')
	for i, snapshot in enumerate(snapshot_selection):
		if isInput:
			node_snapshot = nodes[:, i]
		else:
			node_snapshot = A[node_selection,snapshot]

		if velocity_data is not None:
			velocity_snapshot = velocity_data[:, snapshot]
			snapshot_reconstructed = least_squares_approximation_with_velocity(V, V_simplified, node_snapshot, velocity_snapshot, x_old, timestep)
			snapshot_old = snapshot_reconstructed[node_selection]
		else:
			snapshot_reconstructed = least_squares_approximation(V, V_simplified, node_snapshot)
		
		if snapshot == 0:
			A_reconstructed = snapshot_reconstructed
		else:
			A_reconstructed = np.column_stack((A_reconstructed,snapshot_reconstructed))

		if isCalculateError:
			error_reconstructed_snapshot = np.array(abs(snapshot_reconstructed[:A.shape[0]] - A[: , snapshot]))
			if snapshot == 0:
				error_reconstructed = error_reconstructed_snapshot
			else:
				error_reconstructed = np.column_stack((error_reconstructed,error_reconstructed_snapshot))
		bar.update()
				
	return A_reconstructed, error_reconstructed

def euler(y, dydt, dt):
	return y + dydt*dt

def crank_nicol(y, dydt1, dydt2, dt):
	return y + (dydt1 + dydt2)/2*dt