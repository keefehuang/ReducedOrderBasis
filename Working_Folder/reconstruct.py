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


timestep = 0.008
def reconstruct(full_data_input_file, tracking_nodes_file, simplified_data_file, input_vtk_file, output_vtk_file, mapping_file=None, basis_file=None, isCalculateError=False, static_nodes=None, isVelocityConstraints=False): 
	### Set the parameters for the SVD calculation
	k = int(input("Set the number of basis 'k' for the SVD calculation = "))
	n = int(input("Set the number of power iterations 'n' for the SVD calculation = "))
	params = [k, n]
	
	# Extracting full simulation data
	print("Extracting full simulation data")
	if ".binout" in full_data_input_file:
		coordinates_data, displacement_data, full_data_ids, _, time_data = data_extraction_binout(full_data_input_file, basis_file, isCalculateError, False)
	else:	
		# TODO: a4db formats need to contain time data for error calculation
		coordinates_data, displacement_data, full_data_ids, _ = data_extraction_a4db(full_data_input_file, basis_file, isCalculateError, False)
		time_data = None
	
	# Extracting simplified data
	print("Extracting simplified simulation data")
	
	simplified_time_data = None
	if simplified_data_file.endswith(".npz"):
		_, simplified_displacement_data, simplified_velocity_data, simplified_time_data = data_extraction_npz(simplified_data_file, isVelocityConstraints)
	elif simplified_data_file.endswith(".npy"):
		simplified_displacement_data = data_extraction_npy(simplified_data_file)
	elif ".binout" in simplified_data_file:
		_, simplified_displacement_data, _, simplified_velocity_data, simplified_time_data = data_extraction_binout(simplified_data_file, None, False, True)
	elif ".a4db" in simplified_data_file:
		_, simplified_displacement_data, _, simplified_velocity_data = data_extraction_a4db(full_data_input_file, None, False, isVelocityConstraints)	
	else:
		print("Could not recognize tracking simplified data file... Trying to read objects...")
		simplified_displacement_data = pickle.load(open(simplified_data_file, "rb"))
		print("Read succesfully")


	if isCalculateError:
		if simplified_displacement_data.shape[1] == displacement_data.shape[1]:
			print("No time data is provided, assummed that full data and simplified data match")
		else:
			if time_data is not None and simplified_time_data is not None and displacement_data is not None:	
				displacement_data, simplified_displacement_data, simplified_velocity_data = time_mapping(displacement_data, time_data, simplified_displacement_data, simplified_time_data, simplified_velocity_data)
			else:
				print("Timesteps for simplified data and full data do not match and no time data is provided - error for snapshots cannot be calculated. Error calculation will be turned off")
				isCalculateError = False

	# Extracting tracking point ids and functions for weighting.
	# Note that pkl files cannot store functions
	print("Extracting tracking point ids and weighting functions")
	if tracking_nodes_file.endswith(".py"):
		tracking_nodes_file = ".".join(tracking_nodes_file.split("/")[1:])[:-3]
		tracking_nodes_import = importlib.import_module(tracking_nodes_file)
		tracking_node_ids = tracking_nodes_import.tracking_points
		try:
			functions = tracking_nodes_import.functions
		except:
			functions = None
			print("No functions detected in input")
	elif tracking_nodes_file.endswith(".pkl"):
		tracking_node_ids = pickle.load(open(tracking_nodes_file, "rb"))
		functions = None
		print("No functions detected in input")
	elif tracking_nodes_file.endswith(".npz"):
		import_tracking_nodes = np.load(tracking_nodes_file)
		tracking_node_ids = import_tracking_nodes['tracking_nodes']	
		try:
			functions = import_tracking_nodes['functions']
		except:
			functions = None
			print("No functions detected in input")
	else:
		print("Could not recognize tracking node ids file... Trying to read objects...")
		tracking_node_ids = np.load(tracking_nodes_file)
		functions = None
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
	print("Output Vtks printed")

def main():
	description = "Computes a RB-approximated reconstruction of a Finite Element model"
	epilog = """example:
 	$ python reconstruct.py ./input_data/SFS_MAIN_Full_Model.a4db ./input_data/bumper_data_higher_res.npz ./input_data/testfile.npz ./visualisation/in/Bumper_0.vtk ./visualisation/out/Bumper.vtk -m ./input_data/Bumper_mapping_higher_resolution_a4db.npy -v -e -b ./input_data/basis_vectors.pkl
	""".format("reconstruction.py")

	argparser = argparse.ArgumentParser(description=description,epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
	argparser.add_argument("binoutfile",type=str,metavar="main.binout",help="Results file for full model")
	argparser.add_argument("simplifiednodes",type=str,metavar="simplified.binout",help="Results file for simplified model")
	argparser.add_argument("trackingnodes",type=str,metavar="tracking_node.py",help="Node ids for simplified nodes")
	argparser.add_argument("inputfile",type=str,metavar="original_input.vtk",help="Reference VTK file")
	argparser.add_argument("outputfile",type=str,metavar="reconstructed_output.vtk",help="Output VTK file")
	argparser.add_argument("-m", "--mapping", dest="mappingfile", default=None, type=str, metavar=" ", help="Turns on mapping between main.binout ids and vtk ids")
	argparser.add_argument("-b","--basis", dest="V",default=None,type=str,metavar="ROB.pkl",help="Reduced basis vectors, sorted into xyz, xyz format")
	argparser.add_argument("-s","--static", dest="staticnodes", default=None, type=str, metavar="static_nodes.npy", help="List of static nodes in binout file if applicable")
	# argparser.add_argument("--singleframe",dest="singleframe",nargs="+", default=-1, type=int, metavar="52", help="input frame to be reconstructed")
	argparser.add_argument("-e", "--error",dest="calculateerror", action="store_true", default=False, help="Turns on error calculation")
	argparser.add_argument("-v", "--velocity",dest="isvelocity", action="store_true", default=False, help="Turns on velocity constraints for least square approximations")

	args = argparser.parse_args(args=None if len(sys.argv) > 1 else ['--help'])

	# Call the reconstruction function.
	reconstruct(args.binoutfile, args.trackingnodes, args.simplifiednodes, args.inputfile, args.outputfile, mapping_file=args.mappingfile, basis_file=args.V, isCalculateError=args.calculateerror, isVelocityConstraints=args.isvelocity)

if __name__ == '__main__':
	main()