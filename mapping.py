from Binout_reading import binout_reading
import numpy as np 
import os
import os.path
import glob
import small_func
from qd.cae.dyna import Binout
from writetoVtk import writetoVtk
import numpy as np 
import re
import queue

def res(node1, node2):
	return np.linalg.norm(node1 - node2)

# Creates an array that maps from Binout data to VTK visualisation data
# TODO: somehow dynamically determine the number of VTK datas (probably via some regex)
def vtk_mapping(A, vtk_in_directory, vtk_name, no_binout_nodes, snapshot=1):
	# Total number of nodes 
	no_vtk_nodes = 51391
	no_total_snapshots = A.shape[1]

	binout_state = np.reshape(np.array(A[:,snapshot]), (no_binout_nodes,3))

	updated_variable_name = 'POINTS'
	next_variable_name = 'CELLS'
	vtk = vtk_in_directory + vtk_name + '_' + str(snapshot) + '.vtk'

	with open(vtk, 'r', encoding='utf-8') as vtk_in:
		bodyText = vtk_in.read()
		uv_start 			= 		bodyText.index(updated_variable_name)
		uv_end 				= 		bodyText.index(next_variable_name)
		vtk_state			= 		np.reshape(np.array(list(map(float, bodyText[uv_start + (bodyText[uv_start:].index('\n')):uv_end].strip().split(" ") ))), (no_vtk_nodes,3))

	round_binout_state = np.round(binout_state,decimals=3)
	round_vtk_state = np.round(vtk_state,decimals=3)
	index_map = []

	ind_vtk = np.lexsort((round_vtk_state[:,0], round_vtk_state[:,1], round_vtk_state[:,2]))
	ind_binout = np.lexsort((round_binout_state[:,0], round_binout_state[:,1], round_binout_state[:,2]))


	sorted_vtk = round_vtk_state[ind_vtk,:]
	sorted_binout = round_binout_state[ind_binout,:]	

	for i,node in enumerate(sorted_vtk)	:
		count = 0
		error = 1e-2
		wrong_node = node
		while np.linalg.norm(sorted_binout[i+count, :] - node) > error:
			count += 1

		if(count != 0):
			tmp = sorted_binout[i,:]
			sorted_binout[i,:] = sorted_binout[i+count,:]
			sorted_binout[i+count,:] = tmp

			tmp = ind_binout[i]
			ind_binout[i] = ind_binout[i+count]
			ind_binout[i+count] = tmp

		print("I am in node {}, the error is {}".format(i, np.linalg.norm(binout_state[ind_binout[i]] - sorted_vtk[i])))

		assert np.linalg.norm(binout_state[ind_binout[i]] - sorted_vtk[i]) < 1

	index_map = ind_binout[np.argsort(ind_vtk)]

	matching_binout = binout_state[index_map]

	for i, node in enumerate(vtk_state):
		if np.linalg.norm(matching_binout[i] - node) > 1e-2:
			print("Error at {}, the error is {}".format(i, np.linalg.norm(matching_binout[i] - node)))
			raise NameError("Something went wrong")

	index_map = ind_binout[np.argsort(ind_vtk)]
	
	extended_map = [0]*no_vtk_nodes*3

	for i,node in enumerate(index_map):
		extended_map[3*i] = node*3
		extended_map[3*i+1] = node*3 + 1
		extended_map[3*i+2] = node*3 + 2	

	np.save('extended_map' + vtk_name, extended_map)	

	return 'extended_map' + vtk_name

# maps the selected node data to the Binout data. returns the selected node data and the mapping as an array
def node_mapping(binout_coordinates, simplified_coordinates, simplified_displacements, snapshot = 0, normalize=True):

	num_binout_nodes = binout_coordinates.shape[0]//3
	num_simplified_nodes = simplified_coordinates.shape[0]//3

	index_map = [0]*num_simplified_nodes
	residuals = np.zeros((num_binout_nodes,num_simplified_nodes))

	reference_nodes = np.reshape(np.array(binout_coordinates[:,snapshot]), (num_binout_nodes,3))
	simplified_nodes = np.reshape(np.array(simplified_coordinates[:,snapshot]), (num_simplified_nodes,3))

	round_reference_nodes = np.round(reference_nodes,decimals=3)
	round_simplified_nodes = np.round(simplified_nodes,decimals=3)

	ind_reference_nodes = np.lexsort((round_reference_nodes[:,0], round_reference_nodes[:,1], round_reference_nodes[:,2]))
	ind_simplified_nodes = np.lexsort((round_simplified_nodes[:,0], round_simplified_nodes[:,1], round_simplified_nodes[:,2]))

	sorted_reference_nodes = round_reference_nodes[ind_reference_nodes,:]
	sorted_simplified_nodes = round_simplified_nodes[ind_simplified_nodes,:]

	count = 0
	# first sweep to identify entries that are directly mapped (ie. no averaging was done)
	index_map = []
	simplified_map = []
	mapping_error = []
	residuals = queue.PriorityQueue(maxsize=3)
	error = 1e-2
	for i, node in enumerate(sorted_simplified_nodes):
		print("Working on simplfied node {}".format(i))
		count = 2
		init_residual1 = res(sorted_reference_nodes[0], node)
		init_residual2 = res(sorted_reference_nodes[1], node)
		residuals.put((-init_residual1, 0))
		residuals.put((-init_residual2, 1))
		residual = init_residual1
		while residual > error and count < num_binout_nodes:
			residual = res(sorted_reference_nodes[count], node)
			residuals.put((-residual, count))
			residuals.get()
			count += 1

		if (count >= num_binout_nodes):
			res1 = residuals.get()
			res2 = residuals.get()
			if res1[1] in index_map:
				loc1 = index_map.index(res1[1])
				if -res1[0] < mapping_error[loc1]:
					del index_map[loc1]
					del mapping_error[loc1]
					del simplified_map[loc1]
					index_map.append(res1[1])
					mapping_error.append(-res1[0])
					simplified_map.append(i)
			else:
				index_map.append(res1[1])
				mapping_error.append(-res1[0])
				simplified_map.append(i)

			if res2[1] in index_map:
				loc2 = index_map.index(res2[1])
				if -res2[0] < mapping_error[loc2]:
					del index_map[loc2]
					del mapping_error[loc2]
					del simplified_map[loc2]
					index_map.append(res2[1])
					mapping_error.append(-res2[0])
					simplified_map.append(i)
			else:
				index_map.append(res2[1])
				mapping_error.append(-res2[0])
				simplified_map.append(i)

			# print("Found nodes at {} and {}".format(index_map[-1], index_map[-2]))
		elif(count != 0):
			simplified_map.append(i)
			index_map.append(count-1)
			mapping_error.append(0)
			residuals.get()
			residuals.get()
			# print("Found node at {}".format(index_map[-1]))

	assert np.unique(index_map).shape[0] == len(simplified_map)

	# second sweep to find the closest points

	# # attempt to map nodes to closest coordinates in the Binout file via RMS error stored in the Residual matrix. 
	# # calculate the residual over multiple snapshots to ensure a good matching (unknown effectiveness)
	# while(np.unique(index_map).shape[0] != num_simplified_nodes) and snapshot < 2:
	# 	for j, selected_node in enumerate(simplified_nodes):
	# 		for i, reference_node in enumerate(reference_nodes):
	# 			residuals[i,j] += np.sqrt(np.sum(np.power(selected_node-reference_node,2)))
	# 	index_map = np.argmin(residuals, axis=0)
	# 	snapshot += 1
	# 	# reference_nodes = np.reshape(np.array(binout_coordinates[:,snapshot]), (num_binout_nodes,3))
	# 	# simplified_nodes = np.reshape(np.array(Coordinates[:,snapshot]), (num_simplified_nodes,3))

	# deleted_nodes = np.array([])
	# for i in range(index_map.shape[0]):
	# 	duplicates = np.where(index_map == index_map[i])[0]
	# 	if duplicates.shape[0] > 1:
	# 		res = 10000000
	# 		for j, position in enumerate(duplicates):
	# 			if residuals[position,i] < res:
	# 				res = residuals[position, i]
	# 				index = j
	# 		duplicates = np.delete(duplicates, index)
	# 		deleted_nodes = np.concatenate((deleted_nodes, duplicates))

	# deleted_nodes = np.unique(deleted_nodes)

	# index_map = np.delete(index_map, deleted_nodes)

	extended_map = [0]*len(index_map)*3
	extended_simplified_map = [0]*len(simplified_map)*3
	# extended_deleted_nodes = [0]*deleted_nodes.shape[0]*3

	for i,node in enumerate(index_map):
		extended_map[3*i] = node*3
		extended_map[3*i+1] = node*3 + 1
		extended_map[3*i+2] = node*3 + 2

	# for i, node in enumerate(deleted_nodes):
	# 	extended_deleted_nodes[3*i] = node*3
	# 	extended_deleted_nodes[3*i+1] = node*3 + 1
	# 	extended_deleted_nodes[3*i+2] = node*3 + 2

	for i, node in enumerate(simplified_map):
		extended_simplified_map[3*i] = node*3
		extended_simplified_map[3*i+1] = node*3 + 1
		extended_simplified_map[3*i+2] = node*3 + 2

	# simplified_displacements = np.delete(simplified_displacements, extended_deleted_nodes, axis = 0)
	simplified_displacements = simplified_displacements[extended_simplified_map,:]
	simplified_coordinates = simplified_coordinates[extended_simplified_map,:]
	
	return extended_simplified_map, extended_map

# Takes in the stored array data from readying.py and maps it to Binout data
def preprocessingInputs(dyna_input_name, simplified_node_name):
	# read time, coordinates and displacements from simplified model
	simplified_data = np.load(simplified_node_name)
	simplified_timesteps = simplified_data['Time']
	simplified_coordinates = simplified_data['Coordinates']
	simplified_displacements = simplified_data['Displacements']
	simplified_angular_velocities = simplified_data['AngularRotations']

	# reads time, coordinates and displacements from Binout (ie. full model)
	binout_coordinates, binout_displacements, binout_angular_velocities = binout_reading(dyna_input_name, False, 'coordinates + displacements + rvelocities')
	binout_timesteps = binout_reading(dyna_input_name, False, 'time')

	# small_func rearranges the output matrix A to follow an x-y-z arrangement along index 0
	binout_displacements = small_func.rearange_xyz(binout_displacements)
	binout_coordinates = small_func.rearange_xyz(binout_coordinates)
	binout_angular_velocities = small_func.rearange_xyz(binout_angular_velocities)

	# removes the static nodes (ie nodes that do not move) Movement array stores the displacement over time for all nodes
	# if the displacement is > 1e-12 over the full snapshot, we consider that the node is part of the simulation. Otherwise, we remove 
	# it from consideration when mapping selected node data to Binout data
	Movement = np.zeros((binout_displacements.shape[0],))
	for snapshot in range(1,binout_displacements.shape[1]):
		Movement += np.abs(binout_displacements[:,snapshot] - binout_displacements[:, snapshot-1])

	# Moving displacements/coordinates indicate the nodes which move during the simulation
	binout_moving_indices = np.where(Movement > 1e-12)[0]
	binout_moving_displacements = binout_displacements[binout_moving_indices,:]
	binout_moving_coordinates = binout_coordinates[binout_moving_indices,:]
	binout_moving_angular_velocities = binout_angular_velocities[binout_moving_indices,:]

	# the timesteps of the data from the simplified simulation may not match the timesteps in the binout. This removes all timesteps from
	# the binout data that do not correspond to any simplified data
	time_error = 1e-5
	time_indices = []
	for time_index in simplified_timesteps:
		time_indices.append(np.where(abs(binout_timesteps - time_index) < time_error)[0][0])

	# relevant binout data is referred to as 'moving*'. This indicates that the data matches the timesteps in the simplified simulation
	binout_displacements = binout_displacements[:, time_indices]
	binout_coordinates = binout_coordinates[:, time_indices]
	binout_angular_velocities = binout_angular_velocities[:, time_indices]
	binout_moving_displacements = binout_moving_displacements[:,time_indices]
	binout_moving_coordinates = binout_moving_coordinates[:,time_indices]
	binout_moving_angular_velocities = binout_moving_angular_velocities[:, time_indices]

	print("Starting to map the simplified nodes to the binout nodes!")
	simplified_map, simplified_node_indices = node_mapping(binout_moving_coordinates, simplified_coordinates, simplified_displacements)
	print("Nodes mapped")

	simplified_displacements = simplified_displacements[simplified_map,:]
	simplified_coordinates = simplified_coordinates[simplified_map,:]
	simplified_angular_velocities = simplified_angular_velocities[simplified_map,:]
	
	simplified_variables = (simplified_displacements, simplified_coordinates, simplified_angular_velocities)
	binout_moving_variables = (binout_moving_displacements, binout_moving_coordinates, binout_moving_angular_velocities)
	return simplified_variables, simplified_node_indices, binout_moving_variables, binout_moving_indices, binout_coordinates

# Used to test th e
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

