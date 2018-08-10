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
import copy

# function to calculate 2-norm difference of two nodes
def res(node1, node2):
	return np.linalg.norm(node1 - node2)

# Creates an array that maps from lsdyna data to VTK visualisation data
def vtk_mapping(binout_state, vtk_in_directory, vtk_name="Bumper", no_binout_nodes=52114, snapshot=1):
	# no_total_snapshots = A.shape[1]

	# binout_state = np.reshape(np.array(A[:,snapshot]), (no_binout_nodes,3))

	# Start of the node location data
	updated_variable_name = 'POINTS'
	# End of the node location data
	next_variable_name = 'CELLS'
	# 1st vtk1 (not vtk 0 because structure is different for the first vtk in a series)
	vtk = vtk_in_directory + vtk_name + '_' + str(snapshot) + '.vtk'

	with open(vtk, 'r', encoding='utf-8') as vtk_in:
		bodyText = vtk_in.read()
		# positions before and after node data
		uv_start 			= 		bodyText.index(updated_variable_name)
		uv_end 				= 		bodyText.index(next_variable_name)
		
		# reshaping the data into a #Nodes X 3 matrix
		vtk_state			= 		np.reshape(np.array(list(map(float, bodyText[uv_start + (bodyText[uv_start:].index('\n')):uv_end].strip().split(" ") ))), (-1,3))
		no_vtk_nodes 		=		vtk_state.shape[0]
		
	
	# round_binout_state = np.round(binout_state,decimals=3)
	# round_vtk_state = np.round(vtk_state,decimals=3)
	round_binout_state = binout_state
	round_vtk_state = vtk_state
	index_map = []

	ind_vtk = np.lexsort((round_vtk_state[:,0], round_vtk_state[:,1], round_vtk_state[:,2]))
	ind_binout = np.lexsort((round_binout_state[:,0], round_binout_state[:,1], round_binout_state[:,2]))

	sorted_vtk = round_vtk_state[ind_vtk,:]
	sorted_binout = round_binout_state[ind_binout,:]

	for i,node in enumerate(sorted_vtk):
		count = 0
		error = 3
		switch = True
		while np.linalg.norm(sorted_binout[i+count] - node) > error:
			count += 1
			if (count + i) >= sorted_binout.shape[0]:
				for j, select_node in enumerate(sorted_binout):

					if np.linalg.norm(select_node - node) < error:

						print(ind_binout.shape)
						print(ind_binout[j].reshape(1,).shape)
						sorted_binout = np.concatenate((sorted_binout, sorted_binout[j,:].reshape(1,3)))
						ind_binout = np.concatenate((ind_binout, ind_binout[j].reshape(1,)))
						count = 0
						break

		if(count != 0 and switch):
			tmp = np.copy(sorted_binout[i])
			sorted_binout[i] = sorted_binout[i+count]
			sorted_binout[i+count] = tmp

			tmp =  np.copy(ind_binout[i])
			ind_binout[i] = ind_binout[i+count]
			ind_binout[i+count] = tmp

			# print(np.linalg.norm(sorted_binout[i] - node))
			# assert np.linalg.norm(sorted_binout[i] - node) < error
			# print("Swapping {} with {}\n".format(i, i+count))

		# print("I am in node {}, the error is {}".format(i, np.linalg.norm(binout_state[ind_binout[i]] - sorted_vtk[i])))

		assert np.linalg.norm(binout_state[ind_binout[i]] - sorted_vtk[i]) < error

	index_map = ind_binout[np.argsort(ind_vtk)]

	matching_binout = binout_state[index_map]

	for i, node in enumerate(vtk_state):
		if np.linalg.norm(matching_binout[i] - node) > 1000:
			print("Error at {}, the error is {}".format(i, np.linalg.norm(matching_binout[i] - node)))
			raise NameError("Something went wrong")

	print(np.linalg.norm(matching_binout[i] - node))
	
	extended_map = [0]*no_vtk_nodes*3

	for i,node in enumerate(index_map):
		extended_map[3*i] = node*3
		extended_map[3*i+1] = node*3 + 1
		extended_map[3*i+2] = node*3 + 2	

	np.save('extended_map_abaqus' + vtk_name, extended_map)	

	return 'extended_map' + vtk_name

# maps the selected node data to the Binout data. returns the selected node data and the mapping as an array
def least_squares_node_mapping(binout_coordinates, simplified_coordinates, snapshot = 0, normalize=True):

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
	
	# stores the linkedNodes number of nodes with smallest L2-norm error in a priority queue. 
	index_map = []
	simplified_map = []
	mapping_error = []	
	linkedNodes = 1
	maxsize = linkedNodes + 2
	residuals = queue.PriorityQueue(maxsize=maxsize)
	error = 1e-2
	for i, node in enumerate(sorted_simplified_nodes):
		print("Mapping simplified node {}".format(i))
		count = linkedNodes
		for n in range(linkedNodes):
			init_residual = res(sorted_reference_nodes[n], node)
			residuals.put((-init_residual, n))
		residual = init_residual
		while residual > error and count < num_binout_nodes:
			residual = res(sorted_reference_nodes[count], node)
			residuals.put((-residual, count))
			residuals.get()
			count += 1

		if (count >= num_binout_nodes):
			for n in range(linkedNodes):
				final_residual = residuals.get()
				if final_residual[1] in index_map:
					loc = index_map.index(final_residual[1])
					if -final_residual[0] < mapping_error[loc]:
						del index_map[loc]
						del mapping_error[loc]
						del simplified_map[loc]
						index_map.append(final_residual[1])
						mapping_error.append(-final_residual[0])
						simplified_map.append(i)
				else:
					index_map.append(final_residual[1])
					mapping_error.append(-final_residual[0])
					simplified_map.append(i)
		elif(count != 0):
			simplified_map.append(i)
			index_map.append(count-1)
			mapping_error.append(0)
			for n in range(linkedNodes):
				residuals.get()

	assert np.unique(index_map).shape[0] == len(simplified_map)

	print("Simplified nodes mapped to {} nodes in binout data".format(len(simplified_map)))

	extended_map = [0]*len(index_map)*3
	extended_simplified_map = [0]*len(simplified_map)*3

	for i,node in enumerate(index_map):
		extended_map[3*i] = node*3
		extended_map[3*i+1] = node*3 + 1
		extended_map[3*i+2] = node*3 + 2


	for i, node in enumerate(simplified_map):
		extended_simplified_map[3*i] = node*3
		extended_simplified_map[3*i+1] = node*3 + 1
		extended_simplified_map[3*i+2] = node*3 + 2

	
	return extended_simplified_map, extended_map


# Takes in simplified model data from and maps it to full model data using a least squares calculation
def preprocessing_input_least_squares_mapping(dyna_input_name, simplified_node_name, normalize=True):
	# read time, coordinates and displacements from simplified model
	simplified_data = np.load(simplified_node_name)
	simplified_timesteps 						= 	simplified_data['Time']
	simplified_coordinates_not_normalized 		= 	simplified_data['Coordinates']
	simplified_coordinates 						= 	np.copy(simplified_coordinates_not_normalized)
	simplified_displacements 					=	simplified_data['Displacements']
	simplified_angular_velocities 				=	simplified_data['AngularRotations']

	# reads time, coordinates and displacements from Binout (ie. full model)
	binout_coordinates, binout_displacements, binout_angular_velocities = binout_reading(dyna_input_name, False, 'coordinates + displacements + rvelocities')
	binout_timesteps = binout_reading(dyna_input_name, False, 'time')

	# small_func rearranges the output matrix A to follow an x-y-z arrangement along index 0
	binout_displacements 						= small_func.rearange_xyz(binout_displacements)
	binout_coordinates_not_normalized 			= small_func.rearange_xyz(binout_coordinates)
	binout_coordinates 							= np.copy(binout_coordinates_not_normalized)
	binout_angular_velocities 					= small_func.rearange_xyz(binout_angular_velocities)

	# the timesteps of the data from the simplified simulation may not match the timesteps in the binout. This removes all timesteps from
	# the binout data that do not correspond to any simplified data
	time_error = 1e-5
	time_indices = []
	for time_index in simplified_timesteps:
		# print(time_index)
		# print(np.where(abs(binout_timesteps - time_index) < time_error ) )
		time_indices.append(np.where(abs(binout_timesteps - time_index) < time_error)[0][0])

	# This indicates that the data matches the timesteps in the simplified simulation
	binout_displacements = binout_displacements[:, time_indices]
	binout_coordinates = binout_coordinates[:, time_indices]
	binout_angular_velocities = binout_angular_velocities[:, time_indices]

	if normalize==True:
	    for i in range(0,binout_coordinates.shape[1]):
	        if np.linalg.norm(binout_coordinates[:,i])!=0:
	            binout_coordinates[:,i]				=		binout_coordinates_not_normalized[:,i]/np.linalg.norm(binout_coordinates[:,i])
	            binout_displacements[:,i]			=		binout_displacements[:,i]/np.linalg.norm(binout_coordinates[:,i])
	            # binout_angular_velocities[:,i]		=		binout_angular_velocities[:,i]/np.linalg.norm(binout_coordinates[:,i])
	            simplified_coordinates[:,i]			=		simplified_coordinates_not_normalized[:,i]/np.linalg.norm(binout_coordinates[:,i])
	            simplified_displacements[:,i]		=		simplified_displacements[:,i]/np.linalg.norm(binout_coordinates[:,i])
	            # simplified_angular_velocities[:,i]	=		simplified_angular_velocities[:,i]/np.linalg.norm(binout_coordinates[:,i])

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

	print("Starting to map the simplified nodes to the binout nodes!")
	simplified_map, simplified_node_indices = least_squares_node_mapping(binout_coordinates_not_normalized[binout_moving_indices,:], simplified_coordinates_not_normalized)
	print("Nodes mapped")

	simplified_displacements = simplified_displacements[simplified_map,:]
	simplified_coordinates = simplified_coordinates[simplified_map,:]
	simplified_angular_velocities = simplified_angular_velocities[simplified_map,:]
	
	simplified_variables = (simplified_displacements, simplified_coordinates, simplified_angular_velocities)
	binout_moving_variables = (binout_moving_displacements, binout_moving_coordinates, binout_moving_angular_velocities)
	return simplified_variables, simplified_node_indices, binout_moving_variables, binout_moving_indices, binout_coordinates


# Takes in simplified model data from and maps it to full model data using a given mapping
def preprocessing_input_cross_section_mapping(dyna_input_name, simplified_data_name, node_xsection_name):
	# read time, coordinates and displacements from simplified model
	simplified_data = np.load(simplified_data_name)
	simplified_timesteps 						= 	simplified_data['Time']
	simplified_coordinates 						= 	simplified_data['Coordinates']
	simplified_displacements 					=	simplified_data['Displacements']
	simplified_angular_velocities 				=	simplified_data['AngularRotations']

	simplified_node_num 						= 	simplified_coordinates.shape[0]

	# read in the mapping of cross-sectional node ids to simplified nodes
	node_xsection_data = np.load(node_xsection_name)

	# reads time, coordinates and displacements from Binout (ie. full model)
	binout_coordinates, binout_displacements, binout_angular_velocities = binout_reading(dyna_input_name, False, 'coordinates + displacements + rvelocities')
	binout_timesteps = binout_reading(dyna_input_name, False, 'time')
	

	# small_func rearranges the output matrix A to follow an x-y-z arrangement along index 0
	# TODO: Create function to rearrange binout per node order
	binout_displacements 						= small_func.rearange_xyz(binout_displacements)
	binout_coordinates 							= small_func.rearange_xyz(binout_coordinates)
	binout_angular_velocities 					= small_func.rearange_xyz(binout_angular_velocities)
	Coordinates = binout_coordinates
	

	# the timesteps of the data from the simplified simulation may not match the timesteps in the binout. This removes all timesteps from
	# the binout data that do not correspond to any simplified data
	time_error = 1e-5
	time_indices = []
	for time_index in simplified_timesteps:
		# print(time_index)
		# print(np.where(abs(binout_timesteps - time_index) < time_error ) )
		time_indices.append(np.where(abs(binout_timesteps - time_index) < time_error)[0][0])

	# This indicates that the data matches the timesteps in the simplified simulation
	binout_displacements = binout_displacements[:, time_indices]
	binout_coordinates = binout_coordinates[:, time_indices]
	binout_angular_velocities = binout_angular_velocities[:, time_indices]

	binout_length 								= binout_displacements.shape[1]
	binout_height								= binout_displacements.shape[0]
	binout_moving_indices 						= range(binout_height)

	# averaging the nodes in the main folder
	deleted = 0
	simplified_node_indices = []
	for num in range(simplified_node_num//3):
		print("Processing node {}".format(num+1))
		node_displacements	 		= np.zeros((3, binout_length))
		node_coordinates 			= np.zeros((3, binout_length))
		node_angular_velocities	 	= np.zeros((3, binout_length))
		if node_xsection_data[num] != []:
			for item in node_xsection_data[num]:
				items = [int(item - 1)*3, int(item - 1)*3+1, int(item - 1)*3+2]
				node_displacements = node_displacements + binout_displacements[items,:]
				node_coordinates = node_coordinates + binout_coordinates[items,:]
				node_angular_velocities = node_angular_velocities + binout_angular_velocities[items,:]				

			node_displacements 			= node_displacements/node_xsection_data[num].shape[0]
			node_coordinates 			= node_coordinates/node_xsection_data[num].shape[0]
			node_angular_velocities	 	= node_angular_velocities/node_xsection_data[num].shape[0]	
			nums = [num*3 - deleted, num*3+1 - deleted, num*3+2 - deleted]
			# print("Error:")
			# print(node_coordinates[:,0] - simplified_coordinates[nums,0])
			

			binout_displacements 		= np.concatenate((binout_displacements, node_displacements))
			binout_coordinates 			= np.concatenate((binout_coordinates, node_coordinates))
			binout_angular_velocities	= np.concatenate((binout_angular_velocities, node_angular_velocities))

			simplified_node_indices.append(binout_height)
			simplified_node_indices.append(binout_height + 1)
			simplified_node_indices.append(binout_height + 2)
			binout_height += 3

		else:
			nums = [num*3 - deleted, num*3 + 1 - deleted, num*3 + 2 - deleted]
			simplified_coordinates = np.delete(simplified_coordinates, nums, 0)
			simplified_displacements = np.delete(simplified_displacements, nums, 0)
			simplified_angular_velocities = np.delete(simplified_angular_velocities, nums, 0)
			deleted += 3		



	simplified_variables 	= (simplified_displacements, simplified_coordinates, simplified_angular_velocities)
	binout_variables 		= (binout_displacements, binout_coordinates, binout_angular_velocities)

	# # removes the static nodes (ie nodes that do not move) Movement array stores the displacement over time for all nodes
	# # if the displacement is > 1e-12 over the full snapshot, we consider that the node is part of the simulation. Otherwise, we remove 
	# # it from consideration when mapping selected node data to Binout data
	# Movement = np.zeros((binout_displacements.shape[0],))
	# for snapshot in range(1,binout_displacements.shape[1]):
	# 	Movement += np.abs(binout_displacements[:,snapshot] - binout_displacements[:, snapshot-1])

	# # Moving displacements/coordinates indicate the nodes which move during the simulation
	# binout_moving_indices = np.where(Movement > 1e-12)[0]
	# binout_moving_displacements = binout_displacements[binout_moving_indices,:]
	# binout_moving_coordinates = binout_coordinates[binout_moving_indices,:]
	# binout_moving_angular_velocities = binout_angular_velocities[binout_moving_indices,:]
	

	return simplified_variables, simplified_node_indices, binout_variables, binout_moving_indices, Coordinates[:, time_indices]

# determines the cross-section nodes based off the full model. NOTE THAT THESE DIMENISONS ARE UNIQUE TO THE BUMPER PROBLEM!!!
# Please update boundary conditions to generalize to other problems!!
def simplified_node_xsection_mapping(simplified_data_name = "Bumper_high_resolution_simplified_nodes.npy"):
	# reads time, coordinates and displacements from Binout (ie. full model)
	binout_coordinates = binout_reading("bumper.binout", False, 'coordinates')
	binout_ids = binout_reading("bumper.binout", False, 'ids')
	
	# small_func rearranges the output matrix A to follow an x-y-z arrangement along index 0
	# TODO: Create function to rearrange binout per node order	
	binout_coordinates 							= small_func.rearange_xyz(binout_coordinates)[:,0].reshape((-1,3))

	top_index = (binout_coordinates[:,0] >= 189) & (binout_coordinates[:,0] <= 511) & (binout_coordinates[:,1] >= -1) & (binout_coordinates[:,1] <= 66)
	bot_index = (binout_coordinates[:,0] >= 189) & (binout_coordinates[:,0] <= 511) & (binout_coordinates[:,1] >= 883) & (binout_coordinates[:,1] <= 951)
	left_index = (binout_coordinates[:,0] >= 159) & (binout_coordinates[:,0] <= 191) & (binout_coordinates[:,1] >= -200) & (binout_coordinates[:,1] <= 1150)

	bot 		= binout_coordinates[bot_index]
	bot_ids 	= binout_ids[bot_index]

	top 		= binout_coordinates[top_index]
	top_ids 	= binout_ids[top_index]

	left 		= binout_coordinates[left_index]	
	left_ids	= binout_ids[left_index]

	### reads coordinates of simplified nodes
	simplified_coordinates = np.load(simplified_data_name)

	mapped_nodes = []


	for i,node in enumerate(simplified_coordinates):
		if node[0] < 190:
			xsection_pos =  np.argmin( [res(node[1], xnode) for xnode in left[:,1]] )
			xsection_ids = [res(left[xsection_pos,1], xnode)<1.5 for xnode in left[:,1]]
			xsection = left_ids[xsection_ids]
			
			mapped_nodes.append(xsection)
			
			# print("Left")

		elif node[1] > 885:
			xsection_pos = np.argmin( [ res(node[0], xnode) for xnode in bot[:,0] ] )
			xsection_ids = [res(bot[xsection_pos,0], xnode)<1.5 for xnode in bot[:,0]]
			xsection = bot_ids[xsection_ids]
			mapped_nodes.append(xsection)
			
			#print("Bot")
			


		elif node[1] < 65:
			xsection_pos = np.argmin( [ res(node[0], xnode) for xnode in top[:,0] ] )
			xsection_ids = [res(top[xsection_pos,0], xnode)<2.5 for xnode in top[:,0]]
			xsection = top_ids[xsection_ids]
			mapped_nodes.append(xsection)

			# print("Top")
	for r in mapped_nodes:
		print(sum(1 for node in r))
	np.save("XSection", mapped_nodes)
		



if __name__ == '__main__':
	simplified_node_xsection_mapping()



	# Locate main directory folder to increase portability
	# problem_name = "Bumper"
	# main_directory_path					=		os	.path.dirname(os.path.realpath(__file__))
	# vtk_in_folder = 'Visualization/VTK_IN/'
	# vtk_in_directory = os.path.join(main_directory_path, vtk_in_folder) + problem_name + '/'

	# binout_state = np.load("Full_Node_Data_High_Resolution.npy")
	# binout_state = binout_state[:,1:].reshape(-1,3)


	# vtk_mapping(binout_state, vtk_in_directory)

	#Import lsdyna input file from its own folder: LOOK FOR DYNA INPUT DECK FILES
	# relative_data_path					=		'Data/'  #relative path for dyna-input-files 
	# mask_path							=		main_directory_path
	# input_file_name						=		os.path.join(main_directory_path, relative_data_path)

	# dyna_input_files_available			=		[os.path.basename(x) for x in glob.glob('%s*.binout'%input_file_name)]


	# print('List of dyna input files available in /Data folder:')

	# i = 0
	# for name in dyna_input_files_available:
	#     print('{first} = {second}'.format(first=i, second=name))
	#     i+=1
	    
	# choose_dyna_input_file =  input('choose dyna input file index = ')
	# dyna_input_path        =  os.path.join(main_directory_path, relative_data_path, dyna_input_files_available[int(choose_dyna_input_file)])
	# dyna_input_name        =  dyna_input_files_available[int(choose_dyna_input_file)]
	# total_nodes = 51391
	# A = binout_reading(dyna_input_path, False, 'coordinates')

	# A = small_func.rearange_xyz(A)

	# reduced_nodes = A.shape[0]//3

	# updated_variable_name = 'POINTS'
	# next_variable_name = 'CELLS'
	# vtk_in_folder = 'Visualization/VTK_IN/'
	# vtk_out_folder = 'Visualization/VTK_OUT/'
	# vtk_name = 'Bumper'
	# vtk_in_directory = os.path.join(main_directory_path, vtk_in_folder) + vtk_name + '/'
	# vtk_out_directory = os.path.join(main_directory_path, vtk_out_folder)  + vtk_name + '/'
	# is_static_nodes = True
	# cell_per_row = 9

	# vtk_mapping(A, vtk_in_directory, vtk_name, reduced_nodes, snapshot=1)

