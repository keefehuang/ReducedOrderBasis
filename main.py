from Binout_reading import binout_reading
from ristretto.mf import rsvd
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
from mapping import vtk_mapping
from mapping import node_mapping
from mapping import preprocessing_input_mapping
from reading import read_simplified_data

if __name__ == '__main__':
	
	# Locate main directory folder to increase portability
	main_directory_path					=		os.path.dirname(os.path.realpath(__file__))

	#Import lsdyna input file from its own folder: LOOK FOR DYNA INPUT DECK FILES
	relative_data_path					=		'Data/'  #relative path for dyna-input-files 
	relative_simplified_data_path		=		'Data/Bumper/' #relative path for data from simplified model
	mask_path							=		main_directory_path
	input_file_name						=		os.path.join(main_directory_path, relative_data_path)
	data_input_file_name				=		os.path.join(main_directory_path, relative_simplified_data_path)

	dyna_input_files_available			=		[os.path.basename(x) for x in glob.glob('%s*.binout'%input_file_name)] #dyna files available in provided data folder
	k_input_files_available				=		[os.path.basename(x) for x in glob.glob('%s*.k'%input_file_name)] # input decks available


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
	   
	# This is to take in the original k input file to map LS-DYNA node IDs to binout data.
	choose_k_input_file =  input('choose k input file index (leave blank if not necessary) = ')
	if(choose_k_input_file):
		k_input_path        =	os.path.join(main_directory_path, relative_data_path, k_input_files_available[int(choose_k_input_file)])
		k_input_name        =	k_input_files_available[int(choose_k_input_file)]
	else:
		k_input_name 		=	None

	# location of the vtk input/output in system
	vtk_in_folder = 'Visualization/VTK_IN/'
	vtk_out_folder = 'Visualization/VTK_OUT/'
	# name of the problem - modify this for different problem names
	problem_name = 'Bumper'
	
	vtk_in_directory = os.path.join(main_directory_path, vtk_in_folder) + problem_name + '/'
	vtk_out_directory = os.path.join(main_directory_path, vtk_out_folder)  + problem_name + '/'

	#reads simiplified data in the form of a .npz
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

	# preprocessing maps the simplified data to the binout data
	simplified_variables, simplified_nodes_indices, moving_variables, moving_nodes_indices, Coordinates = preprocessing_input_mapping(dyna_input_name, simplified_data_name, normalize=False)
	simplified_nodes = np.concatenate((simplified_variables[0], simplified_variables[1], simplified_variables[2]), axis = 0)
	moving_displacements = moving_variables[0]
	moving_coordinates =  moving_variables[1]
	moving_angular_velocities =  moving_variables[2]

	# read in snapshot/node parameters
	total_nodes = moving_displacements.shape[0]//3
	total_snapshot = moving_displacements.shape[1]
	
	# Called to perform mapping of Binout data to vtk style for visualisation. Variable mapping_name is the name of the npy storing the
	# mapping from Binout data to LS-DYNA data.
	if not os.path.isfile("extended_map"+problem_name+".npy"):
		mapping_name = vtk_mapping(Coordinates, vtk_in_directory, vtk_name, 51391, snapshot=1)

	mapping_name = "extended_map" + problem_name + ".npy"

##################### Perform rSVD ###########################################################################################
	
	### snapshots in Binout data (column indices) chosen for reconstruction (ie. how many times t to be reconstructed)
	snapshot_selection = range(0,total_snapshot)

	### node selected based on NodeID present in the LS-DYNA mapping data
	# simplified_nodes_indices = readNodes(nodes, total_nodes, k_input_path)

	### generate random set of nodes for testing purposes
	# random_set = randomMask(total_nodes, 600)
	# simplified_nodes_indices = random_set

	# TODO: create an interface for this (??)
	# Manually input name of stored selected node data

	# Ref is the position of all nodes at t0. Added to all reconstructed displacement values to perform visualisation

	ref = Coordinates[:,0].reshape((Coordinates.shape[0],1))
	# Parameters for SDV. Note that required parameters will change depending on the type of SVD performed
	params = [20, 40, 40]

	print("Reconstructing matrix using SVD vectors")

	Variables = (moving_displacements, moving_coordinates, moving_angular_velocities)

	# Reconstruct displacement matrix. See reduced_basis.py for more information
	error_r, Displacements_r = multivariableMatrixReconstruction(Variables, snapshot_selection, simplified_nodes_indices, reducedOrderParams=params, isError=True, isInput=True, nodes=simplified_nodes)	

	# Adding the original coordinates to the reconstructed displacements vector.
	Displacements_full = np.zeros(Coordinates.shape)
	Displacements_full[moving_nodes_indices, :] = Displacements_r
	Coordinates_r = Displacements_full + ref

	Error_full = np.zeros(Coordinates.shape)
	Error_full[moving_nodes_indices,:] = error_r

######################    Printing VTK to Output #############################################################################

is_static_nodes = False
cell_per_row = 3

writetoVtk(Coordinates_r, total_nodes, snapshot_selection, vtk_in_directory, vtk_out_directory, problem_name, cell_per_row, is_static_nodes, mapping_name, isError=True, error_r = Error_full)

