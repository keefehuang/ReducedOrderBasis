import h5py
import numpy as np
from small_func import *
from Binout_reading import binout_reading
from mapping import *
from reduced_order import *
import fbpca
import importlib


## TODO: Extend to include velocity/rotational data! Also need to get time-data in a4db files
def data_extraction_a4db(a4db_name, steps=None, id_array=None):
	### Opens up the a4db file
	a4db 			= 	h5py.File(a4db_name, "r")

	for key in a4db["model_0"]:
		print(key)

	### Extracts the ids from the a4db file
	ids 			= 	np.array(a4db["model_0"]["geometry_0"]["nodes_0"]["grids"]["ids"])
	coordinate_ind 	= 	np.argsort(ids)
	id_data 		= 	ids[coordinate_ind]

	### Extract coordinate data from the a4db, data is sorted prior to concatenation
	coordinates_data = np.array(a4db["model_0"]["geometry_0"]["nodes_0"]["grids"]["coordinates"]["x"])[coordinate_ind]
	node_num = coordinates_data.shape[0]
	coordinates_data = np.concatenate((coordinates_data, np.array(a4db["model_0"]["geometry_0"]["nodes_0"]["grids"]["coordinates"]["y"])[coordinate_ind]))
	coordinates_data = np.concatenate((coordinates_data, np.array(a4db["model_0"]["geometry_0"]["nodes_0"]["grids"]["coordinates"]["z"])[coordinate_ind]))
	coordinates_data = rearange_xyz(coordinates_data)

	steps = len(a4db["model_0"]["results_0"])

	### Extract displacement data from the a4db, data is sorted prior to concatenation
	displacement_data = None
	for step in range(steps):
		s = ('step_%s' % str(step + 1))
		displacements 		= 	np.array(a4db["model_0"]["results_0"][s]["displacements"]["x"])[coordinate_ind]
		displacements 		= 	np.concatenate((displacements, np.array(a4db["model_0"]["results_0"][s]["displacements"]["y"])[coordinate_ind]))
		displacements 		= 	np.concatenate((displacements, np.array(a4db["model_0"]["results_0"][s]["displacements"]["z"])[coordinate_ind]))
		
		if displacement_data is None:
			displacement_data 	= np.empty((3*node_num, steps))
		
		displacement_data[:,step] = displacements

	displacement_data = rearange_xyz(displacement_data)

	print(displacement_data.shape)

	return coordinates_data, [displacement_data], id_data

def data_extraction_binout(binout_name):
	coordinate_data, displacement_data = binout_reading(binout_name, False, 'coordinates + displacements')
	time_data = binout_reading(binout_name, False, 'time')

	coordinate_data = rearange_xyz(coordinate_data)[:,0]

	displacement_data = rearange_xyz(displacement_data)

	id_data	= binout_reading(binout_name, False, 'ids')

	return coordinate_data, [displacement_data], id_data, time_data

def data_extraction_npz(npz_name):
	data = np.load(npz_name)
	coordinate_data = data["Coordinates"]
	displacement_data = data["Displacements"]
	time_data = data["Time"]

	return coordinate_data, [displacement_data], time_data

# def data_extraction_pkl(pkl_name):
# 	with open(pkl_name, 'rb') as f:
#     	data = pickle.load(f)

def data_extraction_npy(npy_name):
	data = np.load(npy_name)

	return data

def data_extraction_py(py_name):
	py_path = py_name[1:].split("/")
	py_path = ".".join(py_path[-4:])[:-3]
	print(py_path)
	data_name = input("Please type the variable name\n")
	data = __import__(py_path, fromlist=[data_name])

	return nodes


if __name__ == '__main__':
	# input_name = "/home/keefe/Documents/BMW/HiWi/Code/ReducedOrderBasis/PortableSVD/Projects/CarCrashModel/Data/SFS_MAIN_NO_ROT.a4db"
	# data_extraction_a4db(input_name)
	# input_name = "/home/keefe/Documents/BMW/HiWi/Code/ReducedOrderBasis/PortableSVD/Projects/CarCrashModel/Data/bumper.binout"
	# data_extraction_binout(input_name)

	input_name = "/home/keefe/Documents/BMW/HiWi/Code/ReducedOrderBasis/PortableSVD/Projects/CarCrashModel/Data/Bumper_data_higher_resolution.npz"
	data_extraction_npz(input_name)
