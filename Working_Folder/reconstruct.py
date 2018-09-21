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
from writetoVtk import *
import pickle
import sys

timestep = 0.01
### Please import the data from a .py. Replace CarCrashModel with project name and nodes with the structure containing the nodes
### Please note that an ITERATIVE data object is required
# from Projects.THUMS.Data.tracking_nodes import tracking_points as tracking_node_ids

def reconstruct(full_data_input_file, tracking_nodes_file, simplified_data_file, input_vtk_file, output_vtk_file, mapping_file=None, singleframe=False, basis_file=None, isCalculateError=False, static_nodes=None, isVelocityUsed=False, functions=None): 
	### Set the parameters for the SVD calculation
	k = int(input("Set the number of basis 'k' for the SVD calculation = "))
	n = int(input("Set the number of power iterations 'n' for the SVD calculation = "))
	params = [k, n]
	
	# Extracting full simulation data
	print("Extracting full simulation data")
	if ".binout" in full_data_input_file:
		coordinates_data, displacement_data, full_data_ids, time_data = data_extraction_binout(full_data_input_file, basis_file, isCalculateError)
	else:	
		coordinates_data, displacement_data, full_data_ids = data_extraction_a4db(full_data_input_file, basis_file, isCalculateError)
			
	print(coordinates_data.shape)
	# Extracting simplified data
	print("Extracting simplified simulation data")
	if simplified_data_file.endswith(".npz"):
		simplified_coordinates_data, simplified_displacement_data, simplified_velocity_data, simplified_time_data = data_extraction_npz(simplified_data_file, isVelocityUsed)
	elif simplified_data_file.endswith(".npy"):
		simplified_displacement_data = data_extraction_npy(simplified_data_file)
	else:
		print("Could not recognize tracking simplified data file... Trying to read objects...")
		simplified_displacement_data = pickle.load(open(simplified_data_file, "rb"))
		print("Read succesfully")

		displacement_data = time_mapping(displacement_data, time_data, simplified_displacement_data, simplified_time_data)

	# Extracting tracking point ids
	print("Extracting tracking point ids")
	if tracking_nodes_file.endswith(".py"):
		tracking_nodes_file = ".".join(tracking_nodes_file.split("/")[1:])[:-3]
		tracking_nodes_import = importlib.import_module(tracking_nodes_file)
		tracking_node_ids = tracking_nodes_import.tracking_points

	elif tracking_nodes_file.endswith(".pkl"):
		tracking_node_ids = pickle.load(open(tracking_nodes_file, "rb"))
	else:
		print("Could not recognize tracking node ids file... Trying to read objects...")
		tracking_node_ids = np.load(tracking_nodes_file)
		print("Read succesfully")

	### Extracting basis vectors
	if basis_file is None:
		print("Calculating basis vectors from full data")
		(V, s , Vt) = fbpca.pca(displacement_data, k=k , n_iter=n)
		print("Storing calculated basis vectors as .pkl")
		pickle.dump(V, open("basis_vectors.pkl", "wb"))
	else:
		# Assumes bases are already rearranged
		print("Extracting reduced basis")	
		if basis_file.endswith(".dat") or basis_file.endswith(".npy"):
			V = np.fromfile(basis_file, dtype=float)
		elif basis_file.endswith(".pkl"):
			V = pickle.load(open(basis_file, "rb"))
		else:
			print("Could not recognize reduced basis file... Trying to read objects...")
			V = pickle.load(open(basis_file, "rb"))
			print("Read succesfully")

	### Storing the dimensions of the various analysed functions
	full_node_num		 	= 	coordinates_data.shape[0]
	timestep_num			=	simplified_displacement_data.shape[1]	
	snapshot_selection 		= 	range(timestep_num)

	### Perform vtk mapping as necessary
	if mapping_file is None:
		mapping_file, static_nodes = vtk_mapping(coordinates_data, input_vtk_file, output_vtk_file, full_node_num)	

	### Xsection mapping averages the xsection nodes to create new nodes for SVD
	V, simplified_node_indices = append_tracking_point_rows(V, full_data_ids, tracking_node_ids, functions)

	### Reconstruction of matrix based on SVD

	A_r, error_r = reduced_order_approximation(V, snapshot_selection, simplified_node_indices, isCalculateError=isCalculateError, nodes=simplified_displacement_data, isInput=True, A=displacement_data, timestep=timestep, velocity_data=simplified_velocity_data)
	
	### Adding the initial coordinates to the reconstructed displacements
	A_r = A_r[:full_node_num,:] + coordinates_data.reshape((-1,1))

	### Appending static nodes to reconstructed matrix A
	if static_nodes is not None:
		static_nodes = np.tile(np.load(static_nodes).reshape((-1,1)), (1,len(snapshot_selection)))
		A_r = np.concatenate((A_r, static_nodes))

	### Output the reconstructed data
	writetoVtk(A_r, full_node_num, snapshot_selection, input_vtk_file, output_vtk_file, mapping_file, isCalculateError=isCalculateError, error_r=error_r)

def main():
	description = "Computes a RB-approximated reconstruction of a Finite Element model"
	epilog = """example:
	$ python3 reconstruct.py ../Excluded_01_SoftHandsModel/mor_Thums.k.binout0000 THUMS_Positioning/tracking_nodes.py  THUMS_Positioning/simplified_displacement_data.npy   ../02_initial_inputdeck/THUMS_V.k ./THUMS_V_recon.k2H
	$ {0} Binout/sim.binout Tracking_Nodes/tn.py Target_Position/tn.npy  sim.key sim_reconstructed.key -b V_disp.pkl

	notes: - """.format("reconstruction.py")

	argparser = argparse.ArgumentParser(description=description,epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
	argparser.add_argument("binoutfile",type=str,metavar="main.binout",help="LS-DYNA Binout results file")
	argparser.add_argument("trackingnodes",type=str,metavar="tracking_node_definition.py",help="definition of the tracking nodes")
	argparser.add_argument("simplifiednodes",type=str,metavar="target_position.pkl",help="final position of tracking nodes for reconstruction")
	argparser.add_argument("inputfile",type=str,metavar="original_input.vtk",help="reference VTK file for mapping")
	argparser.add_argument("outputfile",type=str,metavar="reconstructed_output.vtk",help="reference VTK file for data output")
	argparser.add_argument("-m", "--mapping", dest="mappingfile", default=None, type=str, metavar="mapping between input file type to output vtk used for visualisation")
	argparser.add_argument("-b","--basis", dest="V",default=None,type=str,metavar="ROB.pkl",help="reduced basis matrix.")
	argparser.add_argument("-s","--static", dest="staticnodes", default=None, type=str, metavar="static_nodes.npy", help="list of static nodes if appropriate")
	argparser.add_argument("--singleframe",dest="singleframe",nargs="+", default=-1, type=int, metavar="52", help="input frame to be reconstructed")
	argparser.add_argument("-e", "--error",dest="calculateerror", action="store_true", default=False, help="turns on error calculation")
	argparser.add_argument("-v", "--velocity",dest="isvelocity", action="store_true", default=False, help="turns on velocity")
	argparser.add_argument("-f", "--functions", dest="functionfile", default=None, type=str, metavar="functions to calcualte tracking node positions")
	#argparser.set_defaults(singleframe=True)

	args = argparser.parse_args(args=None if len(sys.argv) > 1 else ['--help'])

	# Call the reconstruction function.
	reconstruct(args.binoutfile, args.trackingnodes, args.simplifiednodes, args.inputfile, args.outputfile, mapping_file=args.mappingfile, singleframe=args.singleframe, basis_file=args.V, isCalculateError=args.calculateerror, isVelocityUsed=args.isvelocity, functions=args.functionfile)

if __name__ == '__main__':
	main()