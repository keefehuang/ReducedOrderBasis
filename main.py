from Binout_reading import binout_reading
from ristretto.mf import rsvd
from mask import randomMask
from mask import readNodes
from reduced_order import matrixReconstruction
from reduced_order import matrixReconstructionWithVelocity
from writetoVtk import writetoVtk
import numpy as np 
import scipy as sci
import small_func
import glob
import os.path
import re
from mapping import vtk_mapping
from mapping import node_mapping
from preprocessing import preprocessingInputs

if __name__ == '__main__':
	
	# Locate main directory folder to increase portability
	main_directory_path					=		os.path.dirname(os.path.realpath(__file__))

	#Import lsdyna input file from its own folder: LOOK FOR DYNA INPUT DECK FILES
	relative_data_path					=		'Data/'  #relative path for dyna-input-files 
	mask_path							=		main_directory_path
	input_file_name						=		os.path.join(main_directory_path, relative_data_path)

	dyna_input_files_available			=		[os.path.basename(x) for x in glob.glob('%s*.binout'%input_file_name)]
	k_input_files_available				=		[os.path.basename(x) for x in glob.glob('%s*.k'%input_file_name)]

	print('List of dyna input files available in ' + relative_data_path + ' folder:')
	i = 0
	for name in dyna_input_files_available:
	    print('{first} = {second}'.format(first=i, second=name))
	    i+=1
	    
	choose_dyna_input_file =  input('choose dyna input file index = ')
	dyna_input_path        =  os.path.join(main_directory_path, relative_data_path, dyna_input_files_available[int(choose_dyna_input_file)])
	dyna_input_name        =  dyna_input_files_available[int(choose_dyna_input_file)]

	print('List of input deck files available in ' + relative_data_path + ' folder:')

	i = 0
	for name in k_input_files_available:
	    print('{first} = {second}'.format(first=i, second=name))
	    i+=1
	   
	# This is to take in the orignial k input file to map LS-DYNA node IDs to binout data.
	choose_k_input_file =  input('choose k input file index (leave blank if not necessary) = ')
	if(choose_k_input_file):
		k_input_path        =	os.path.join(main_directory_path, relative_data_path, k_input_files_available[int(choose_k_input_file)])
		k_input_name        =	k_input_files_available[int(choose_k_input_file)]
	else:
		k_input_name 		=	None

	# location of the vtk in system
	vtk_in_folder = 'Visualization/VTK_IN/'
	vtk_out_folder = 'Visualization/VTK_OUT/'
	problem_name = 'Bumper'
	selected_node_data = 'Bumper_Data.npz'
	vtk_in_directory = os.path.join(main_directory_path, vtk_in_folder) + problem_name + '/'
	vtk_out_directory = os.path.join(main_directory_path, vtk_out_folder)  + problem_name + '/'

	# TODO: Add some form of options when running program.

	nodes, selected_nodes_indices, Coordinates, Displacements = preprocessingInputs(dyna_input_name, selected_node_data)

	print(nodes.shape)
	print(selected_nodes_indices.shape)

	# Displacements, Coordinates, Velocities = binout_reading(dyna_input_name, True, 'coordinates + displacements + velocities')
	Coordinates, Displacements = binout_reading(dyna_input_name, False, 'coordinates + displacements')

	# read in snapshot/node parameters
	total_nodes = Displacements.shape[0]//3
	total_snapshot = Displacements.shape[1]
	

	# TODO: Somehow streamline this process (Does this take too long? How do we improve this?)
	# Called to perform mapping of Binout data to vtk style for visualisation. Variable mapping_name is the name of the npy storing the
	# mapping from Binout data to LS-DYNA data.
	# mapping_name = mapping(A, vtk_in_directory, vtk_name, total_nodes, snapshot=1)

	# Manually created mapping folder:
	mapping_name = "extended_mapBumper.npy"


##################### Perform rSVD ###########################################################################################
	

	# snapshots in Binout data (column indices) chosen for reconstruction (ie. how many times t to be reconstructed)
	snapshot_selection = range(0,total_snapshot)

	# nodes in Binout data (row indices) for reconstruction (ie. how many nodes used for reconstruction)

	# node selected based on NodeID present in the LS-DYNA mapping data
	# selected_nodes_indices = readNodes(nodes, total_nodes, k_input_path)

	# generate random set of nodes for testing purposes
	# random_set = randomMask(total_nodes,300)
	selected_nodes_indices = range(0,total_nodes)

	# TODO: create an interface for this
	# Manually input name of stored selected node data

	# Ref is the position of all nodes at t0. Added to all reconstructed displacement values to perform visualisation

	ref = Coordinates[:,0].reshape((Coordinates.shape[0],1))
	# Parameters for SDV. Note that required parameters will change depending on the type of SVD performed
	params = [200, 4, 40]

	# Reconstruct displacement matrix. See reduced_basis.py for more information
	error_r, Displacements_r = matrixReconstruction(Displacements, snapshot_selection, selected_nodes_indices, reducedOrderParams=params, isError=True, nodes=nodes)	

	# Reconstruct displacement matrix. See reduced_basis.py for more information
	# error_r, Displacements_r, Velocities_r = matrixReconstructionWithVelocity(Displacements, Velocities, snapshot_selection, selected_nodes_indices, isError=True)	

	# Adding the original coordinates to the reconstructed displacements vector.
	Displacements_r = Displacements_r + ref


######################    Printing VTK to Output #############################################################################

is_static_nodes = True
cell_per_row = 3

writetoVtk(Displacements_r, total_nodes, snapshot_selection, vtk_in_directory, vtk_out_directory, vtk_name, cell_per_row, is_static_nodes, isError=True, error_r = error_r)

