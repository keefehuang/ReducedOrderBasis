from Binout_reading import binout_reading
from fbpca import pca
from mask import randomMask
from mask import readNodes
from reduced_order import matrixReconstruction
from reduced_order import multivariableMatrixReconstruction
from writetoVtk import writetoVtk
import numpy as np 
import scipy as sci
import small_func
import glob
import os.path
import re
from mapping import *

if __name__ == '__main__':
	
	### Locate main directory folder to increase portability
	main_directory_path					=		os.path.dirname(os.path.realpath(__file__))

	### Import lsdyna input file from its own folder: LOOK FOR DYNA INPUT DECK FILES
	relative_data_path					=		'Data/'  #relative path for dyna-input-files 
	relative_simplified_data_path		=		'Data/Bumper/' #relative path for data from simplified model
	relative_node_xsection_path			=		'XSectionNodes/Original/'  #relative path for node xsection files 
	
	mask_path							=		main_directory_path
	input_file_name						=		os.path.join(main_directory_path, relative_data_path)
	data_input_file_name				=		os.path.join(main_directory_path, relative_simplified_data_path)
	node_xsection_file_name				= 		os.path.join(main_directory_path, relative_node_xsection_path)

	dyna_input_files_available			=		[os.path.basename(x) for x in glob.glob('%s*.binout'%input_file_name)] #dyna files available in provided data folder
	k_input_files_available				=		[os.path.basename(x) for x in glob.glob('%s*.k'%input_file_name)] # input decks available


	### read in full model simulation data - this simulation should include all data used for the rsvd reconstruction
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
	   
	### read in input deck, this is used to map IDs from the simplified model to the full model
	choose_k_input_file =  input('choose k input file index (leave blank if not necessary) = ')
	if(choose_k_input_file):
		k_input_path        =	os.path.join(main_directory_path, relative_data_path, k_input_files_available[int(choose_k_input_file)])
		k_input_name        =	k_input_files_available[int(choose_k_input_file)]
	else:
		k_input_name 		=	None

	### location of the vtk input/output for visualisation
	vtk_in_folder = 'Visualization/VTK_IN/'
	vtk_out_folder = 'Visualization/VTK_OUT/'

	### name of the problem - modify this for different problem names
	problem_name = 'Bumper'
	
	vtk_in_directory = os.path.join(main_directory_path, vtk_in_folder) + problem_name + '/'
	vtk_out_directory = os.path.join(main_directory_path, vtk_out_folder)  + problem_name + '/'

	### reads simiplified node data in the form of a .npz
	isReading = input("Do you need to read in new data?[T/F]\n")
	if isReading == 'T':
		isReading = 1
	if isReading == 'F':
		isReading = 0

	while(isReading != 0 and isReading != 1):
		print("Incorrect input, please enter value again")
		isReading = input("Do you need to read in new data?[T/F]\n")
		if isReading == 'T':
			isReading = 1
		if isReading == 'F':
			isReading = 0

	if isReading:
		simplified_data_name = read_simplified_data(data_input_file_name, problem_name)
	else:
		simplified_data_name = 'Bumper_data.npz'
	
	print("Pulling data from simplified model")

	### maps the simplified nodes to the full model nodes via a least squares calculation
	# simplified_variables, simplified_nodes_indices, moving_variables, moving_nodes_indices, Coordinates = preprocessing_input_least_squares_mapping(dyna_input_name, simplified_data_name, normalize=False)
	
	### maps the simplified nodes to the full model nodes via a provided mapping
	
	### node_xsection_variables indicates the location of the pre-made mapping stored as a XSectionNodes file
	node_xsection_name   		= 		os.path.join(node_xsection_file_name, 'XSectionNodes.npy')
	simplified_variables, simplified_nodes_indices, moving_variables, moving_nodes_indices, Coordinates = preprocessing_input_cross_section_mapping(dyna_input_name, simplified_data_name, node_xsection_name)
	# preprocessing_input_cross_section_mapping(dyna_input_name, simplified_data_name, node_xsection_name):
	

	### Various combinations of Displacements, Velocity, Coordinates for the rsvd reconstruction

	simplified_nodes = np.concatenate((simplified_variables[0], simplified_variables[2]), axis = 0)
	# simplified_nodes = np.concatenate((simplified_variables[0], simplified_variables[2]), axis = 0)
	# simplified_nodes = simplified_variables[0]

	### stores the total number of nodes and number of time-steps in the pre-processed data
	total_nodes = moving_variables[0].shape[0]//3
	total_snapshot = moving_variables[0].shape[1]
	moving_variables = [moving_variables[0], moving_variables[2]]
	
	### Called to perform mapping of Binout data to vtk style for visualisation. Variable mapping_name is the name of the npy storing the
	### mapping from Binout data to LS-DYNA data.
	if not os.path.isfile("extended_map_lsdyna"+problem_name+".npy"):
		mapping_name = vtk_mapping(Coordinates, vtk_in_directory, vtk_name, 51391, snapshot=1)

	mapping_name = "extended_map_lsdyna" + problem_name + ".npy"

##################### Perform rSVD ###########################################################################################
	
	### snapshots in Binout data (column indices) chosen for reconstruction (ie. how many times t to be reconstructed)
	snapshot_selection = range(0,total_snapshot)

	### node selected based on NodeID present in the LS-DYNA mapping data
	# simplified_nodes_indices = readNodes(nodes, total_nodes, k_input_path)

	### generate random set of nodes for testing purposes
	# random_set = randomMask(total_nodes, 600)
	# simplified_nodes_indices = random_set

	### TODO: create an interface for this (??)
	### Manually input name of stored selected node data

	### Ref is the position of all nodes at t0. Added to all reconstructed displacement values
	initial_coordinates = Coordinates[:,0].reshape((Coordinates.shape[0],1))
	
	### Parameters for SDV. Note that required parameters will change depending on the type of SVD performed
	params = [4, 4, 30]

	print("Reconstructing matrix using SVD vectors")	

	### Reconstruct displacement matrix using a single variable. See reduced_basis.py for more information
	# error_r, Displacements_r = matrixReconstruction(moving_variables[0], snapshot_selection, simplified_nodes_indices, reducedOrderParams=params, isError=True, isInput=True, nodes=simplified_nodes)	

	### Reconstruct displacement matrix using multiple variables. See reduced_basis.py for more information
	error_r, Displacements_r = multivariableMatrixReconstruction(moving_variables, snapshot_selection, simplified_nodes_indices, reducedOrderParams=params, isError=True, isInput=True, nodes=simplified_nodes)	

	error_full= error_r[:Coordinates.shape[0],:]
	Displacements_full = Displacements_r[:Coordinates.shape[0],:]

	### Adding the original coordinates to the reconstructed displacements vector.
	Coordinates_r = Displacements_full + initial_coordinates


	Error_full = np.zeros(Coordinates.shape)
	Error_full[moving_nodes_indices,:] = error_full

######################    Printing VTK to Output #############################################################################

is_static_nodes = False
cell_per_row = 3


writetoVtk(Coordinates_r, total_nodes, snapshot_selection, vtk_in_directory, vtk_out_directory, problem_name, cell_per_row, is_static_nodes, mapping_name, isError=True, error_r = Error_full)

