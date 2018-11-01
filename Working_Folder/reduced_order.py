import numpy as np
from scipy.optimize import least_squares
try:
	import pyprind
except:
	pass

scalingfactor = 0

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
	return np.matmul(V_simplified,x) - rhs_simplified + scalingfactor * abs((rhs_old - np.matmul(V_simplified,x))/timestep - velocity_simplified)

# This function solves a least squares optimization problem with given cost function defined in "cost_function".
# The base problem is given as V_simplified * x_simplified = rhs_simplified
# The cost function attempts to penalize unphysical changes in velocity
def least_squares_approximation_with_velocity(V, V_simplified, rhs_simplified, velocity, rhs_old, timestep):
	if rhs_old is None:
		rhs_approx, res, rank, s = np.linalg.lstsq(V_simplified, rhs_simplified, rcond=None)
	else:
		rhs = least_squares(cost_function, np.zeros(V_simplified.shape[1]), args=(rhs_old, rhs_simplified, velocity, V, V_simplified, timestep))
		rhs_approx = rhs.x
	rhs_reconstructed = V.dot(rhs_approx)

	return rhs_reconstructed

# Performs a for loop over all snapshots in Binout data "A" and calls least_squares_approximation which runs the
# least squares approximation for a single snapshot
def reduced_order_approximation(V, node_selection, isCalculateError=False, nodes=None, A=None, velocity_data=None, timestep=None):
	error_reconstructed = None
	snapshot_selection = range(nodes.shape[1])
	V_simplified = V[node_selection, :]
	if velocity_data is not None:
		# velocity_simplified = velocity_data[node_selection, :]
		reconstruction_title = "Reconstructing snapshots with velocity"
		x_old = None
		v_snapshot = None
		v_old = None
	else:
		reconstruction_title = "Reconstructing snapshots"

	try:
		bar = pyprind.ProgBar(len(snapshot_selection), monitor=True, title=reconstruction_title, bar_char='█')
	except:
		pass
	for i, snapshot in enumerate(snapshot_selection):
		node_snapshot = nodes[:, i]			
		if velocity_data is not None:
			snapshot_reconstructed = least_squares_approximation_with_velocity(V, V_simplified, node_snapshot, v_old, x_old, timestep)
			x_old = snapshot_reconstructed[node_selection]
			if v_old is not None:
				v_snapshot = (v_old + velocity_data[:, snapshot]) * 0.5
			v_old = velocity_data[:, snapshot]
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
		try:
			bar.update()
		except:
			pass
				
	return A_reconstructed, error_reconstructed

def euler(y, dydt, dt):
	return y + dydt*dt

def crank_nicol(y, dydt1, dydt2, dt):
	return y + (dydt1 + dydt2)/2*dt