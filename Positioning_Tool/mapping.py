# Add files in the parent folder to the path
import sys
import os
from os.path import splitext
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from Binout_reading import binout_reading
import numpy as np 
import small_func
from qd.cae.dyna import Binout
from preprocessing import *
import numpy as np 
import re
import queue
import copy
import sys

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

### Generates and concatenates new basis vectors using the tracking node ids. Returns updated input data with new appended rows
### and the indices in the updated input data related to the simplified data nodes
def append_tracking_point_rows(full_data, full_data_ids, tracking_point_nodes, weights=None):	
	
	if len(full_data.shape) == 1:
		full_data = full_data.reshape((-1,1))
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

		
if __name__ == '__main__':
	args = sys.argv
	full_input_file_path 		= args[1]
	vtk_input_file_path		= args[2]
	full_coordinates_data, full_data, full_data_ids, time_data = data_extraction_binout(full_input_file_path)

	vtk_mapping(full_coordinates_data, vtk_input_file_path, full_data[0].shape[0])