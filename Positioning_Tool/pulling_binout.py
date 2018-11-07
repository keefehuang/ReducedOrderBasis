import numpy as np
from small_func import *
from Binout_reading import binout_reading
from mapping import *
from reduced_order import *
import fbpca
import importlib
from preprocessing import *
from writetoOutput import *
import pickle
import sys

if __name__ == '__main__':
	binout_file = "./bumper.binout"
	target_position_file = None
	basis_file = None
	


	full_coordinates_data, full_data, full_data_ids, time_data = data_extraction_binout(binout_file, basis_file, target_position_file)
	
	tracking_nodes_file = "./testfile.npz"
	import_tracking_nodes = np.load(tracking_nodes_file)
	tracking_node_ids = import_tracking_nodes['tracking_nodes']	

	# print(full_data.shape[:,180:])
	np.save("full_data", full_data[:,180:])
	
	target_data, tracking_ids = append_tracking_point_rows(full_data, full_data_ids, tracking_node_ids)
	target_data	   = target_data[tracking_ids,:]		

	target_data_name = "target_data"
	target_ids_name = "target_ids"

	

	np.save(target_data_name, target_data[:,180:])
	np.save(target_ids_name, tracking_ids)

