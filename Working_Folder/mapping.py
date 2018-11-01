# Add files in the parent folder to the path
import sys
import os
from os.path import splitext
import argparse
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# Library imports
import numpy as np 
import os.path
import re
import copy
import importlib
try:
	import pyprind
except:
	pass

# weighted average performed on tracking points
def weighted_average(tracking_points, weights):
	if weights is None:
		return simple_average(tracking_points)
	else:
		summed_points = None
		for i, point in enumerate(range(tracking_points.shape[0]//3)):
			if summed_points is None:
				summed_points = weights[i]*tracking_points[[point*3, point*3+1, point*3+2], :]
			else:
				summed_points += weights[i]*tracking_points[[point*3, point*3+1, point*3+2], :]
		return np.array(summed_points/(tracking_points.shape[0]/3))
# simple average performed on tracking points
def simple_average(tracking_points):
	summed_points = None
	for point in range(tracking_points.shape[0]//3):
		if summed_points is None:
			summed_points = tracking_points[[point*3, point*3+1, point*3+2], :]
		else:
			summed_points += tracking_points[[point*3, point*3+1, point*3+2], :]
	return summed_points/(tracking_points.shape[0]/3)


# function to calculate 2-norm difference of two nodes
def res(node1, node2):
	return np.linalg.norm(node1 - node2)

# Creates an array that maps from lsdyna data to VTK visualisation data
def vtk_mapping(full_data_state, input_vtk_file, mapping_directory, full_node_num ):
	full_data_state = np.reshape(np.array(full_data_state), (-1,3))

	# Start of the node location data
	updated_variable_name = 'POINTS'
	# End of the node location data
	next_variable_name = 'METADATA'

	with open(input_vtk_file, 'r', encoding='utf-8') as vtk_in:
		bodyText = vtk_in.read()
		# positions before and after node data
		uv_start 			= 		bodyText.index(updated_variable_name)
		uv_end 				= 		bodyText[uv_start:].index(next_variable_name) + uv_start
		
		# reshaping the data into a #Nodes X 3 matrix
		vtk_state			= 		np.reshape(np.array(list(map(float, bodyText[uv_start + (bodyText[uv_start:].index('\n')):uv_end].strip().split(" ") ))), (-1,3))
		num_vtk_nodes 		=		vtk_state.shape[0]
		
	round_binout_state = np.round(full_data_state,decimals=3)
	round_vtk_state = np.round(vtk_state,decimals=3)
	index_map = []

	ind_binout = np.argsort(round_binout_state[:,0])
	ind_vtk = np.argsort(round_vtk_state[:,0])
	
	sorted_binout = full_data_state[ind_binout,:]
	sorted_vtk = vtk_state[ind_vtk,:]
	
	num_static_nodes = 0
	static_nodes = None
	
	print("Mapping simulation nodes to vtk nodes")
	try:
		bar = pyprind.ProgBar(sorted_vtk.shape[0], monitor=True, title='Mapping Progress', bar_char='█')
	except:
		pass
	for i,node in enumerate(sorted_vtk):
		count = 0
		error = 1e-2
		while np.linalg.norm(sorted_binout[i+count] - node) > error:
			count += 1			
			if (count + i) >= sorted_binout.shape[0]:				
				for j, select_node in enumerate(sorted_binout):
					if np.linalg.norm(select_node - node) < error:
						sorted_binout = np.concatenate((sorted_binout, sorted_binout[j,:].reshape((1,3))))
						ind_binout = np.concatenate((ind_binout, ind_binout[j].reshape(1,1)))
						count = 0
						break
				if count != 0:
					print("Static Node detected, # of static nodes {}".format(num_static_nodes))
					num_static_nodes += 1
					ind_bount = np.concatenate((ind_binout, full_node_num + num_static_nodes))
					if static_nodes is None:
						static_nodes = node
					else:
						static_nodes = np.concatenate((static_nodes, node).reshape((1,3)))
					count = 0
					break

		if(count != 0):
			tmp = np.copy(sorted_binout[i])
			sorted_binout[i] = sorted_binout[i+count]
			sorted_binout[i+count] = tmp

			tmp =  np.copy(ind_binout[i])
			ind_binout[i] = ind_binout[i+count]
			ind_binout[i+count] = tmp

		if ind_binout[i] < full_node_num:
			assert np.linalg.norm(full_data_state[ind_binout[i]] - sorted_vtk[i]) < error
		else:
			assert np.linalg.norm(static_nodes[num_static_nodes] - sorted_vtk[i]) < error
		try:
			bar.update(item_id = "Node {}/{}".format(i, sorted_vtk.shape[0]))
		except:
			pass

	index_map = ind_binout[np.argsort(ind_vtk)]
	if static_nodes is not None:
		full_data_state = np.concatenate((full_data_state, static_nodes.reshape((-1,3))))
	matching_binout = full_data_state[index_map]

	for i, node in enumerate(vtk_state):
		if np.linalg.norm(matching_binout[i] - node) > error:
			print("Error at {}, the error is {}".format(i, np.linalg.norm(matching_binout[i] - node)))
			raise ValueError("Something went wrong")
	
	extended_map = [0]*num_vtk_nodes*3

	for i,node in enumerate(index_map):
		extended_map[3*i] = node*3
		extended_map[3*i+1] = node*3 + 1
		extended_map[3*i+2] = node*3 + 2	

	mapping_save_file		= "./mapping.npy"
	np.save(mapping_save_file, extended_map)	
	print("Mapping file saved to " + mapping_save_file)
	static_nodes_save_file = None
	if static_nodes is not None:
		static_nodes_save_file 	= "static_nodes.npy"
		np.save(static_nodes_save_file, static_nodes)	
		print("Static nodes saved to " + static_nodes_save_file)
	return mapping_save_file, static_nodes_save_file


### Generates and concatenates new basis vectors using the tracking node ids. Returns updated input data with new appended rows
### and the indices in the updated input data related to the simplified data nodes
def append_tracking_point_rows(full_data, full_data_ids, tracking_point_nodes, weights=None):	
	
	tracking_node_indices = []
	tracking_point_nodes = np.array(tracking_point_nodes)
	tracking_point_num = tracking_point_nodes.shape[0]

	full_data_height = full_data.shape[0]
	full_data_length = full_data.shape[1]

	for num in range(tracking_point_num):
		node_data	 		= None
		if np.array(tracking_point_nodes[num]).shape[0] != 1:
			for node_id in tracking_point_nodes[num]:
				# node position refers to the location of the node id in the full_data_ids array
				node_id_position = np.where(full_data_ids == node_id)[0][0]
				# degrees_of_freedom refer to the position of the degrees of freedom associated with node id in node_position
				node_id_degrees_of_freedom = [node_id_position*3, node_id_position*3+1, node_id_position*3+2]
				if node_data is None:
					node_data = full_data[node_id_degrees_of_freedom,:]
				else:	
					node_data = np.concatenate((node_data, full_data[node_id_degrees_of_freedom,:]))

			if weights is not None:
				full_data = np.concatenate((full_data, weighted_average(node_data, weights[num])))
			else:
				full_data = np.concatenate((full_data, simple_average(node_data)))
			  
			tracking_node_indices.append(full_data_height)
			tracking_node_indices.append(full_data_height + 1)
			tracking_node_indices.append(full_data_height + 2)
			full_data_height += 3
		else:
			node_id_position = np.where(full_data_ids == tracking_point_nodes[num])[0][0]
			tracking_node_indices.append(node_id_position*3)
			tracking_node_indices.append(node_id_position*3+1)
			tracking_node_indices.append(node_id_position*3+2)
	print("Mapping Complete")
	return full_data, tracking_node_indices
		