from Binout_reading import binout_reading
import numpy as np 
import os
import os.path
import glob
import small_func
from qd.cae.dyna import Binout
from sorting import *
import numpy as np 
import re
import queue
import copy

# function to calculate 2-norm difference of two nodes
def res(node1, node2):
	return np.linalg.norm(node1 - node2)

# Creates an array that maps from lsdyna data to VTK visualisation data
def vtk_mapping(binout_state, vtk_in_directory, mapping_directory, vtk_name="Bumper", no_binout_nodes=52114, snapshot=0):
	binout_state = np.reshape(np.array(binout_state), (-1,3))

	# Start of the node location data
	updated_variable_name = 'POINTS'
	# End of the node location data
	next_variable_name = 'METADATA'
	# 1st vtk1 (not vtk 0 because structure is different for the first vtk in a series)
	vtk = vtk_in_directory + vtk_name + '_' + str(snapshot) + '.vtk'

	with open(vtk, 'r', encoding='utf-8') as vtk_in:
		bodyText = vtk_in.read()
		# positions before and after node data
		uv_start 			= 		bodyText.index(updated_variable_name)
		uv_end 				= 		bodyText[uv_start:].index(next_variable_name) + uv_start
		
		# reshaping the data into a #Nodes X 3 matrix
		vtk_state			= 		np.reshape(np.array(list(map(float, bodyText[uv_start + (bodyText[uv_start:].index('\n')):uv_end].strip().split(" ") ))), (-1,3))
		no_vtk_nodes 		=		vtk_state.shape[0]
		
	
	round_binout_state = np.round(binout_state,decimals=3)
	round_vtk_state = np.round(vtk_state,decimals=3)
	index_map = []

	ind_vtk = np.argsort(round_vtk_state[:,0])
	ind_binout = np.argsort(round_binout_state[:,0])

	sorted_vtk = round_vtk_state[ind_vtk,:]
	sorted_binout = round_binout_state[ind_binout,:]

	print("Mapping simulation nodes to vtk nodes")
	for i,node in enumerate(sorted_vtk):
		count = 0
		error = 1e-2
		switch = True
		while np.linalg.norm(sorted_binout[i+count] - node) > error:
			count += 1
			if (count + i) >= sorted_binout.shape[0]:				
				for j, select_node in enumerate(sorted_binout):
					if np.linalg.norm(select_node - node) < error:
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

		assert np.linalg.norm(binout_state[ind_binout[i]] - sorted_vtk[i]) < error

	index_map = ind_binout[np.argsort(ind_vtk)]

	matching_binout = binout_state[index_map]

	for i, node in enumerate(vtk_state):
		if np.linalg.norm(matching_binout[i] - node) > 1000:
			print("Error at {}, the error is {}".format(i, np.linalg.norm(matching_binout[i] - node)))
			raise NameError("Something went wrong")
	
	extended_map = [0]*no_vtk_nodes*3

	for i,node in enumerate(index_map):
		extended_map[3*i] = node*3
		extended_map[3*i+1] = node*3 + 1
		extended_map[3*i+2] = node*3 + 2	

	save_location = os.path.join(mapping_directory, 'extended_map_' + vtk_name + '.npy')
	np.save(save_location, extended_map)	

	return save_location


def xsection_mapping(full_data, full_data_ids, xsection_data, simplified_node_num, full_node_num, timestep_num):
	simplified_node_indices = []
	node_height = full_node_num

	for i, data_set in enumerate(full_data):
		print("Processing data set {}".format(i+1))
		for num in range(simplified_node_num):
			node_data	 		= np.zeros((3, timestep_num))		
			if xsection_data[num] != []:
				for item in xsection_data[num]:
					path = np.where(full_data_ids == item)[0][0]
					items = [path*3, path*3+1, path*3+2]
					node_data = node_data + data_set[items,:]			

				node_data 			= node_data/xsection_data[num].shape[0]
				data_set 			= np.concatenate((data_set, node_data))

				simplified_node_indices.append(node_height)
				simplified_node_indices.append(node_height + 1)
				simplified_node_indices.append(node_height + 2)
				node_height += 3
			else:
				try:
					path = np.where(full_data_ids == xsection_data[num][0])[0][0]
					simplified_node_indices.append(path*3)
					simplified_node_indices.append(path*3+1)
					simplified_node_indices.append(path*3+2)
				except:
					pass
		full_data[i] = data_set
	return full_data, simplified_node_indices


def time_mapping(full_data, time_data, simplified_data, simplified_time_data):
	time_error = 1e-5
	time_indices = []

	simplified_time_data = simplified_time_data * 10
	for time_index in simplified_time_data:
		time_indices.append(np.where(abs(time_data - time_index) < time_error)[0][0])


	for i, data_set in enumerate(full_data):
		data_set = data_set[:,time_indices]
		full_data[i] = data_set
	return full_data
		