# Add files in the parent folder to the path
import sys
import os
from os.path import splitext
import argparse
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# Library imports
import numpy as np
from small_func import *
from Binout_reading import binout_reading
from mapping import *
from reduced_order import *
import fbpca
import importlib
from preprocessing import *
from writetoOutput import *
import pickle
import sys

### Please import the data from a .py. Replace CarCrashModel with project name and nodes with the structure containing the nodes
### Please note that an ITERATIVE data object is required

def reconstruct(binout_file, tracking_nodes_file, keyfile, output_file, singleframe=True, basis_file=None, target_position_file=None):
	
	### Set the parameters for the SVD calculation
	n = 5	
	k = 5
	
	if isinstance(basis_file, (list, tuple)):
		basis_file = basis_file[0]

	### Extracting data from files
	print("Extracting full simulation data")
	if basis_file is not None and target_position_file is not None:
		full_coordinates_data, full_data_ids = data_extraction_binout(binout_file, basis_file, target_position_file)
		full_data = None
	else:
		print('No reduced basis has been provided or no target position has been provided. Gathering all snapshots...')
		full_coordinates_data, full_data, full_data_ids, time_data = data_extraction_binout(binout_file, basis_file, target_position_file)

	### Stores orignal number of basis vectors
	full_node_num = full_coordinates_data.shape[0]

	# Extracting tracking point ids and weights
	print("Extracting tracking point ids and weighting functions")
	if tracking_nodes_file.endswith(".py"):
		tracking_nodes_file = ".".join(tracking_nodes_file.split("/")[1:])[:-3]
		tracking_nodes_import = importlib.import_module(tracking_nodes_file)
		tracking_node_ids = tracking_nodes_import.tracking_points
		try:
			weights = tracking_nodes_import.weights
		except:
			weights = None
			print("No weights detected in input")
	elif tracking_nodes_file.endswith(".pkl"):
		tracking_node_data = pickle.load(open(tracking_nodes_file, "rb"))
		try:
			tracking_node_ids = tracking_node_data[0]
			weights = tracking_node_data[1]
		except:
			weights = None
			tracking_node_ids = tracking_node_data
			print("No weights detected in input")
	elif tracking_nodes_file.endswith(".npz"):
		import_tracking_nodes = np.load(tracking_nodes_file)
		tracking_node_ids = import_tracking_nodes['tracking_nodes']	
		try:
			weights = import_tracking_nodes['weights']
		except:
			functions = None
			print("No weights detected in input")
	else:
		print("Could not recognize tracking node ids file... Trying to read objects...")
		tracking_node_ids = np.load(tracking_nodes_file)	
		weights = None
		print("Read succesfully")

	### Extracting target position data
	print("Extracting target position data")
	if target_position_file is not None:
		if target_position_file[-3:] == "npy":
			target_data = np.load(target_position_file)[:,-1]
		elif target_position_file[-3:] == "pkl":
			target_data = pickle.load(open(target_position_file, "rb"))
		elif target_position_file.endswith(".binout"):
			target_data = binout_reading(target_position_file)
			target_data = target_data['displacements']
		else:
			print("Could not recognize target position input file... Trying to read objects...")
			target_data = pickle.load(open(target_position_file, "rb"))
	else:
		print("Target position calculated from last snapshot of binout")

		target_data, tracking_ids = append_tracking_point_rows(full_data[:,-1].reshape((-1,1)), full_data_ids, tracking_node_ids)
		target_data	   = target_data[tracking_ids,:]		

	### Extracting basis vectors
	if basis_file is None:
		print("Calculating basis vectors from full data")
		(V, s , Vt) = fbpca.pca(full_data, k=k , n_iter=n, raw=True)
		with open("basis_vectors.pkl", "wb") as f:
			pickle.dump(V, f)
	else:
		### Assumes bases are already rearranged
		print("Extracting reduced basis")	
		if basis_file[-3:] == "dat" or basis_file[-3:] == "npy":
			V = np.fromfile(basis_file, dtype=float)
		elif basis_file[-3:] == "pkl":
			with open(basis_file, "rb") as f:
				V = pickle.load(f)
		else:
			with open(basis_file, "rb") as f:
				V = pickle.load(f)
	
	### Xsection mapping averages the tracking_nodes nodes to create new nodes for SVD	
	V, target_node_indices = append_tracking_point_rows(V, full_data_ids, tracking_node_ids, weights)

	### Reconstruction of matrix based on SVD
	A_r = reducedOrderApproximation(V, target_node_indices, nodes=target_data)
	
	### Adding the initial coordinates to the reconstructed displacements
	A_r = full_coordinates_data + A_r[:full_node_num,:]

	print("Writing to Output File")
	### Output the reconstructed data
	writetoOutput(keyfile, output_file, A_r, full_data_ids)

def main():
	description = "Computes a RB-approximated reconstruction of a Finite Element model"
	epilog = """example:
	 # $ python reconstruct.py ./sample.binout ./tracking_nodes.binout ./input.key ./output.key -t ./tracking_points
	 $ {0} Binout/sim.binout Tracking_Nodes/tn.py Target_Position/tn.npy  sim.key sim_reconstructed.key -b V_disp.pkl
	 
	 notes: - """.format("reconstruct.py")

	argparser = argparse.ArgumentParser(description=description,epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
	argparser.add_argument("binoutfile",type=str,metavar="main.binout",help="LS-DYNA Binout results file")
	argparser.add_argument("trackingnodes",type=str,metavar="tracking_node_definition.py",help="Definition of the tracking nodes")
	argparser.add_argument("keyfile",type=str,metavar="original_output.key",help="LS-DYNA input keyword file which includes the node definitions")
	argparser.add_argument("outputfile",type=str,metavar="reconstructed_output.key",help="Output keyword file which the reconstructed output will be written to")

	argparser.add_argument("-b","--basis",dest="V",default=None,type=str,metavar="basis_vectors.pkl",help="Reduced basis matrix.")
	# argparser.add_argument("--singleframe",dest="singleframe", default=-1, type=int, metavar="int", help="Input frame to be reconstructed")
	argparser.add_argument("-t", "--target",dest="targetposition", default=None, metavar="targetposition.pkl", help="Final positions of tracking nodes")
	#argparser.set_defaults(singleframe=True)
	
	args = argparser.parse_args(args=None if len(sys.argv) > 1 else ['--help'])
	
	# Call the reconstruction function.
	reconstruct(args.binoutfile, args.trackingnodes, args.keyfile, args.outputfile, basis_file=args.V, target_position_file=args.targetposition)

if __name__ == '__main__':
	main()
