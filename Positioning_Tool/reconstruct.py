# Add files in the parent folder to the path
import sys
import os
from os.path import splitext
import argparse
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# Library imports
import numpy as np
import pickle
import sys
import fbpca
import importlib

#Local imports
from mapping import *
from reduced_order import *
from preprocessing import *
from writetoOutput import *
from classes import *

def reconstruct(full_data_input_file, tracking_nodes_file, keyfile, output_file, singleframe=True, basis_file=None, target_position_file=None):
	
	n = 5	
	k = 5
	
	if isinstance(basis_file, (list, tuple)):
		basis_file = basis_file[0]

	print("Extracting full simulation data")
	full_data_input = Input(full_data_input_file, basis_file, target_position_file)
	coordinates_data, displacement_data, full_id_data, time_data = full_data_input.extract_main()
	# ### Stores orignal number of basis vectors
	full_node_num = coordinates_data.shape[0]

	print("Extracting tracking point ids and weighting functions")
	tracking_node_data = Input(tracking_nodes_file, basis_file)
	tracking_id_data, tracking_node_list, weights = tracking_node_data.extract_tracking_points_and_weights()
	sort_tracking = np.argsort(tracking_id_data)
	tracking_id_data = tracking_id_data[sort_tracking]
	tracking_node_list = tracking_node_list[sort_tracking]

	print("Extracting target position data")
	if target_position_file is not None:
		target_data_input = Input(target_position_file, basis_file)
		target_data, target_id_data, target_velocity_data, target_time_data =\
		target_data_input.extract_simple()
		sort_target = np.argsort(target_id_data)
		target_data = target_data.reshape((-1,3))
		target_data = target_data[sort_target,:]
		target_data = target_data.reshape((-1,1))
	else:
		target_data, tracking_ids = append_tracking_point_rows(displacement_data[:,-1].reshape((-1,1)), full_id_data, tracking_node_list)
		target_data	   = target_data[tracking_ids,:]
		target_pkl_data = [tracking_id_data, target_data]
		print("Pickling target data")
		with open(output_file.rsplit(".",1)[0] + "_td.pkl", "wb") as f:
			pickle.dump(target_pkl_data, f)

	if basis_file is None:
		print("Calculating basis vectors from full data")
		(V, s , Vt) = fbpca.pca(displacement_data, k=k , n_iter=n, raw=True)
		print("Storing calculated basis vectors as .pkl")
		with open("basis_vectors.pkl", "wb") as f:
			pickle.dump(V, f)
	else:
		# Assumes bases are already rearranged
		print("Extracting reduced basis")	
		basis_file_data = Input(basis_file, None, False, False)
		V = basis_file_data.extract_basis()
	### Xsection mapping averages the tracking_nodes nodes to create new nodes for SVD	
	V, target_node_indices = append_tracking_point_rows(V, full_id_data, tracking_node_list, weights)
	### Reconstruction of matrix based on SVD
	A_r = reducedOrderApproximation(V, target_node_indices, target_data, coordinates_data)
	
	output_keyfile = Input(keyfile, basis_file, target_position_file)

	# if full_data_input.dataTypeHandle.type == "key":
	inputdeck = output_keyfile.dataTypeHandle.get_deck()
	# inputdeck.modify_nodes(full_id_data, A_r)
	inputdeck.modify_nodes(full_id_data, A_r)
	inputdeck.write_inputdeck(newfilename=output_file)
	print("Writing to Output File")

def main():
	description = "Computes a RB-approximated reconstruction of a Finite Element model"
	epilog = """example:
	 # $ python reconstruct.py sample.binout tracking_nodes.binout input.key output.key -t tracking_points -b basis_file
	 # $ python reconstruct.py open_source_sim/main.k.binout0000 thums_tracking.pkl open_source_sim/01_inputs/main.k thums_test.k -t modified_thums_td.pkl -b basis_vectors.pkl
	 notes: - """.format("reconstruct.py")

	argparser = argparse.ArgumentParser(description=description,epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
	argparser.add_argument("binoutfile",type=str,metavar="main.key",help="LS-DYNA input deck")
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
