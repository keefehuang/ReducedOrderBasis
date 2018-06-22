from Binout_reading import binout_reading
from functools import partial
import numpy as np 
import os
import os.path
import glob
import small_func
from qd.cae.dyna import Binout


# Creates an array that maps from Binout data to VTK visualisation data
# TODO: somehow dynamically determine the number of VTK datas (probably via some regex)
def vtk_mapping(A, vtk_in_directory, vtk_name, no_binout_nodes, snapshot=1):
	# Total number of nodes 
	no_vtk_nodes = 51391
	no_total_snapshots = A.shape[1]
	# print(A.shape)

	binout_state = np.reshape(np.array(A[:,snapshot]), (no_binout_nodes,3))


	updated_variable_name = 'POINTS'
	next_variable_name = 'CELLS'
	vtk = vtk_in_directory + vtk_name + '_' + str(snapshot) + '.vtk'

	with open(vtk, 'r', encoding='utf-8') as vtk_in:
		bodyText = vtk_in.read()
		uv_start 			= 		bodyText.index(updated_variable_name)
		uv_end 				= 		bodyText.index(next_variable_name)
		vtk_state			= 		np.reshape(np.array(list(map(float, bodyText[uv_start + (bodyText[uv_start:].index('\n')):uv_end].strip().split(" ") ))), (no_vtk_nodes,3))

	index_map = []
	
	for i, node in enumerate(vtk_state):
		print("I am in node " + str(i))
		minres = 1000000000
		index = 0
		for j, ref in enumerate(binout_state):
			res = np.sum(np.sqrt(np.power(np.abs(node-ref), 2)))
			if res < minres:
				minres = res
				index = j
		index_map.append(index)	

	extended_map = [0]*no_binout_nodes*3

	for i,node in enumerate(index_map):
		extended_map[3*i] = node*3
		extended_map[3*i+1] = node*3 + 1
		extended_map[3*i+2] = node*3 + 2

	np.save('extended_map' + vtk_name, extended_map)	

	return 'extended_map' + vtk_name

# maps the selected node data to the Binout data. returns the selected node data and the mapping as an array
def node_mapping(referenceCoordinates, selected_node_name, snapshot = 0, normalize=True):

	selected_data = np.load(selected_node_name)
	Displacements = selected_data['Displacements']
	Coordinates = selected_data['Coordinates']
	Time = selected_data['Time']

	no_reference_nodes = referenceCoordinates.shape[0]//3
	no_selected_nodes = Coordinates.shape[0]//3

	index_map = [0]*no_selected_nodes
	residuals = np.zeros((no_reference_nodes,no_selected_nodes))


	# attempt to map nodes to closest coordinates in the Binout file via RMS error stored in the Residual matrix. 
	# calculate the residual over multiple snapshots to ensure a good matching (unknown effectiveness)
	reference_nodes = np.reshape(np.array(referenceCoordinates[:,snapshot]), (no_reference_nodes,3))
	selected_nodes = np.reshape(np.array(Coordinates[:,snapshot]), (no_selected_nodes,3))
	while(len(set(index_map)) != len(selected_nodes)) and snapshot < 5:
		for j, selected_node in enumerate(selected_nodes):
			for i, reference_node in enumerate(reference_nodes):
				residuals[i,j] += np.sqrt(np.sum(np.power(selected_node-reference_node,2)))
		index_map = np.argmin(residuals, axis=0)
		snapshot += 1
		reference_nodes = np.reshape(np.array(referenceCoordinates[:,snapshot]), (no_reference_nodes,3))
		selected_nodes = np.reshape(np.array(Coordinates[:,snapshot]), (no_selected_nodes,3))

	# Attempt to remove nodes that are too close to other nodes (ie cause overlaps)
	deleted_nodes = np.array([])
	for i in range(index_map.shape[0]):
		duplicates = np.where(index_map == index_map[i])[0]
		# print(duplicates)
		if duplicates.shape[0] > 1:
			res = 10000000
			for j, position in enumerate(duplicates):
				if residuals[position,i] < res:
					res = residuals[position, i]
					index = j
			duplicates = np.delete(duplicates, index)
			deleted_nodes = np.concatenate((deleted_nodes, duplicates))

	deleted_nodes = np.unique(deleted_nodes)

	index_map = np.delete(index_map, deleted_nodes)

	extended_map = [0]*index_map.shape[0]*3
	extended_deleted_nodes = [0]*deleted_nodes.shape[0]*3

	for i,node in enumerate(index_map):
		extended_map[3*i] = node*3
		extended_map[3*i+1] = node*3 + 1
		extended_map[3*i+2] = node*3 + 2

	for i, node in enumerate(deleted_nodes):
		extended_deleted_nodes[3*i] = node*3
		extended_deleted_nodes[3*i+1] = node*3 + 1
		extended_deleted_nodes[3*i+2] = node*3 + 2

	Displacements = np.delete(Displacements, extended_deleted_nodes, axis = 0)
	
	return Displacements, extended_map, Time


if __name__ == '__main__':

	# Locate main directory folder to increase portability
	main_directory_path					=		os	.path.dirname(os.path.realpath(__file__))

	#Import lsdyna input file from its own folder: LOOK FOR DYNA INPUT DECK FILES
	relative_data_path					=		'Data/'  #relative path for dyna-input-files 
	mask_path							=		main_directory_path
	input_file_name						=		os.path.join(main_directory_path, relative_data_path)

	dyna_input_files_available			=		[os.path.basename(x) for x in glob.glob('%s*.binout'%input_file_name)]


	print('List of dyna input files available in /Data folder:')

	i = 0
	for name in dyna_input_files_available:
	    print('{first} = {second}'.format(first=i, second=name))
	    i+=1
	    
	choose_dyna_input_file =  input('choose dyna input file index = ')
	dyna_input_path        =  os.path.join(main_directory_path, relative_data_path, dyna_input_files_available[int(choose_dyna_input_file)])
	dyna_input_name        =  dyna_input_files_available[int(choose_dyna_input_file)]
	total_nodes = 51391
	A = binout_reading(dyna_input_path, False, 'coordinates')

	A = small_func.rearange_xyz(A)

	# infoMatrices = np.load("Bumper_Data.npz")

	# A = infoMatrices["Coordinates"]

	reduced_nodes = A.shape[0]//3

	updated_variable_name = 'POINTS'
	next_variable_name = 'CELLS'
	vtk_in_folder = 'Visualization/VTK_IN/'
	vtk_out_folder = 'Visualization/VTK_OUT/'
	vtk_name = 'Bumper'
	vtk_in_directory = os.path.join(main_directory_path, vtk_in_folder) + vtk_name + '/'
	vtk_out_directory = os.path.join(main_directory_path, vtk_out_folder)  + vtk_name + '/'
	is_static_nodes = True
	cell_per_row = 9

	vtk_mapping(A, vtk_in_directory, vtk_name, reduced_nodes, snapshot=1)


