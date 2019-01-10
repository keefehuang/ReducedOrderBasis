from preprocessing import *
from Binout_reading import *
from small_func import *
from position_tool_inputdeck_writer import *
import numpy as np
import importlib
import sys

if sys.version_info[0] < 3:
	import cPickle as pickle
else:
	import pickle

try:
	from Binout_reading import binout_reading
	isQD = True
except:
	print("qd.cae library cannot be loaded, loading from Binout format is not supported in this python instance")
	isQD = False
try:
	import h5py
	ish5py = True
except:
	print("h5py library cannot be loaded, loading from a4db format is not supported in this python instance")
	ish5py = False


class Input:
	def __init__(self, name, isBasis=False, isCalculateError=False, isVelocity=False):
		# try:
		if isCalculateError is None:
			isCalculateError = True
		try:
			self.dataTypeHandle = inputTypes[name.split(".")[-1]](name, isBasis, isCalculateError, isVelocity)
		except:
			if '.binout' in name:
				self.dataTypeHandle = Binout_input(name, isBasis, isCalculateError, isVelocity)

	def extract_main(self):
		return self.dataTypeHandle.extract_main()

	def extract_simple(self):
		return self.dataTypeHandle.extract_simple()

	def extract_tracking_points_and_weights(self):
		return self.dataTypeHandle.extract_tracking_points_and_weights()

	def extract_basis(self):
		return self.dataTypeHandle.extract_basis()


class a4db_input:
	def __init__(self, name, isBasis, isCalculateError, isVelocity):
		self.type = "a4db"
		self.a4db = h5py.File(name, "r")
		self.steps = len(self.a4db["model_0"]["results_0"])
		self.node_num = node_num = np.array(self.a4db["model_0"]["geometry_0"]["nodes_0"]["grids"]["coordinates"]["x"]).shape[0]
		### Extracts the ids from the self.a4db file. Sorts the ids beforehand to
		ids = np.array(self.a4db["model_0"]["geometry_0"]["nodes_0"]["grids"]["ids"])
		self.coordinate_ind = np.argsort(ids)
		self.ids = ids[self.coordinate_ind]

		self.isExtractFullData = isBasis is None or isCalculateError or isCalculateError is None
		self.isVelocity = isVelocity

	def get_type(self):
		return self.type

	def extract_main(self):

		### Extract coordinate data from the self.a4db, data is sorted prior to concatenation
		coordinate_data = np.array(self.a4db["model_0"]["geometry_0"]["nodes_0"]["grids"]["coordinates"]["x"])[self.coordinate_ind]
		coordinate_data = np.concatenate((coordinate_data, np.array(self.a4db["model_0"]["geometry_0"]["nodes_0"]["grids"]["coordinates"]["y"])[self.coordinate_ind]))
		coordinate_data = np.concatenate((coordinate_data, np.array(self.a4db["model_0"]["geometry_0"]["nodes_0"]["grids"]["coordinates"]["z"])[self.coordinate_ind]))
		coordinate_data = rearange_xyz(coordinate_data)
		coordinate_data = coordinate_data.reshape((-1,1))

		### Extract displacement data from the self.a4db, data is sorted prior to concatenation
		displacement_data = None

		if self.isExtractFullData:
			displacement_data 	= np.empty((3*self.node_num, self.steps))
			for step in range(self.steps):
				s = ('step_%s' % str(step + 1))
				displacements 		= 	np.array(self.a4db["model_0"]["results_0"][s]["displacements"]["x"])[self.coordinate_ind]
				displacements 		= 	np.concatenate((displacements, np.array(self.a4db["model_0"]["results_0"][s]["displacements"]["y"])[self.coordinate_ind]))
				displacements 		= 	np.concatenate((displacements, np.array(self.a4db["model_0"]["results_0"][s]["displacements"]["z"])[self.coordinate_ind]))
				
				displacement_data[:,step] = displacements

			displacement_data = rearange_xyz(displacement_data)
	
		return coordinate_data, displacement_data, self.ids, None

	def extract_simple(self):
		simple_id_indices = np.where((self.ids < 58000000) & (self.ids > 0))
		if self.isExtractFullData:
			simplified_displacement_data 	= np.empty((3*self.node_num, self.steps))
			for step in range(self.steps):
				s = ('step_%s' % str(step + 1))
				displacements 		= 	np.array(self.a4db["model_0"]["results_0"][s]["displacements"]["x"])[self.coordinate_ind]
				displacements 		= 	np.concatenate((displacements, np.array(self.a4db["model_0"]["results_0"][s]["displacements"]["y"])[self.coordinate_ind]))
				displacements 		= 	np.concatenate((displacements, np.array(self.a4db["model_0"]["results_0"][s]["displacements"]["z"])[self.coordinate_ind]))
				
				simplified_displacement_data[:,step] = displacements

			simplified_displacement_data = rearange_xyz(simplified_displacement_data)

		simplified_velocity_data = None
		
		if self.isVelocity:
			simplified_velocity_data 	= np.empty((3*self.node_num, self.steps))
			for step in range(self.steps):
				s = ('step_%s' % str(step + 1))
				velocities 		= 	np.array(self.a4db["model_0"]["results_0"][s]["function_6"]['node']['function'])[self.coordinate_ind]
				velocities 		= 	np.concatenate((velocities, np.array(self.a4db["model_0"]["results_0"][s]["function_7"]['node']['function'])[self.coordinate_ind]))
				velocities 		= 	np.concatenate((velocities, np.array(self.a4db["model_0"]["results_0"][s]["function_8"]['node']['function'])[self.coordinate_ind]))
								
				simplified_velocity_data[:,step] = velocities

			simplified_velocity_data = rearange_xyz(simplified_velocity_data)		

		return simplified_displacement_data, self.ids[simple_id_indices], simplified_velocity_data, None

	def extract_tracking_points_and_weights(self):
		raise TypeError("This is not possible with " + self.type + " data type")

	def extract_basis(self):
		raise TypeError("This is not possible with " + self.type + " data type")


class Binout_input:
	def __init__(self, name, isBasis, isCalculateError, isVelocity):
		self.type = "binout"
		self.name = name;	
		self.isExtractFullData = isBasis is None or isCalculateError or isVelocity is None
		self.isVelocity = isVelocity

	def get_type(self):
		return self.type

	def extract_main(self):
		if self.isExtractFullData:
			coordinate_data, displacement_data = binout_reading(self.name, False, 'coordinates + displacements')
			displacement_data = rearange_xyz(displacement_data)
		else:
			coordinate_data = binout_reading(self.name, False, 'coordinates')
			displacement_data = None

		time_data = binout_reading(self.name, False, 'time')
		coordinate_data = rearange_xyz(coordinate_data)[:,0].reshape((-1,1))
		id_data	= binout_reading(self.name, False, 'ids')

		return coordinate_data, displacement_data, id_data, time_data

	def extract_simple(self):

		simplified_displacement_data = binout_reading(self.name, False, 'displacements')
		simplified_displacement_data = rearange_xyz(simplified_displacement_data)
		simplified_id_data	= binout_reading(self.name, False, 'ids')
		simplified_time_data = binout_reading(self.name, False, 'time')

		if self.isVelocity:
			simplified_velocity_data = binout_reading(self.name, False, 'velocities')
			simplified_velocity_data = rearange_xyz(simplified_velocity_data)
		else:
			simplified_velocity_data = None

		return simplified_displacement_data, simplified_id_data, simplified_velocity_data, simplified_time_data

	def extract_tracking_points_and_weights(self):
		raise TypeError("This is not possible with " + self.type + " data type")

	def extract_basis(self):
		raise TypeError("This is not possible with " + self.type + " data type")

class npz_input:
	def __init__(self, name, isBasis, isCalculateError, isVelocity):
		self.type = "npz"
		self.data = np.load(name)
		self.isExtractFullData = isBasis is None or isCalculateError or isCalculateError is None
		self.isVelocity = isVelocity

	def get_type(self):
		return self.type

	def extract_main(self):		
		id_data = self.data["ids"]
		coordinate_data = self.data["Coordinates"].reshape((-1,1))
		
		if self.isExtractFullData:
			displacement_data = self.data["Displacements"]
		else:
			displacement_data = None
		try:
			time_data = self.data["Time"]
		except:
			time_data = None

		return coordinate_data, displacement_data, id_data, time_data


	def extract_simple(self):	
		simplified_displacement_data = self.data["Displacements"]
		try:
			simplified_id_data = self.data["ids"]
		except:
			simplified_id_data = None
		try:
			simplified_time_data = self.data["Time"]
		except:
			time_data = None
		if self.isVelocity:
			simplified_velocity_data = self.data["Coordinates"]
		else:
			simplified_velocity_data = None

		return simplified_displacement_data, simplified_id_data, simplified_velocity_data, simplified_time_data

	def extract_tracking_points_and_weights(self):
		try:
			weights = self.data['weights']
		except:
			weights = None
			print("No weights detected in input")
		return self.data['tracking_node_ids'], self.data['tracking_nodes'], weights

	def extract_basis(self):
		return self.data['basis_vectors']

class pkl_input:
	def __init__(self, name, isBasis, isCalculateError, isVelocity):
		with open(name, "rb") as f:
			self.data = pickle.load(f)

	def get_type(self):
		return self.type

	def extract_main(self):
		raise TypeError("This is not possible with " + self.type + " data type")

	def extract_simple(self):
		return self.data[1], self.data[0], None, None

	def extract_tracking_points_and_weights(self):
		return np.array(self.data[0]), np.array(self.data[1]), np.array(self.data[2])

	def extract_basis(self):
		return self.data

class py_input:
	def __init__(self, name, isBasis, isCalculateError, isVelocity):
		self.type = "py"
		tracking_nodes_file = ".".join(name.split("/")[1:])[:-3]
		tracking_nodes_import = importlib.import_module(tracking_nodes_file)
		self.tracking_node_ids = tracking_nodes_import.tracking_points
		try:
			self.weights = tracking_nodes_import.weights
		except:
			self.weights = None
			print("No weights detected in input")

	def get_type(self):
		return self.type

	def extract_main(self):
		raise TypeError("This is not possible with " + self.type + " data type")

	def extract_simple(self):
		raise TypeError("This is not possible with " + self.type + " data type")

	def extract_tracking_points_and_weights(self):
		return self.tracking_node_ids, np.array(self.weights)

	def extract_basis(self):
		raise TypeError("This is not possible with " + self.type + " data type")


class k_input:
	def __init__(self, name, isBasis, isCalculateError, isVelocity):
		self.type = "key"
		self.inputdeck = inputdeck(name)		

	def get_type(self):
		return self.type

	def extract_main(self):
		coordinate_data, id_data, _ = self.inputdeck.get_nodes()
		coordinate_data = coordinate_data.reshape((-1,1))
		displacement_data = None
		time_data = None

		return coordinate_data, displacement_data, id_data, time_data

	def extract_simple(self):
		raise TypeError("This is not possible with " + self.type + " data type")

	def extract_tracking_points_and_weights(self):
		raise TypeError("This is not possible with " + self.type + " data type")

	def extract_basis(self):
		raise TypeError("This is not possible with " + self.type + " data type")

	def get_deck(self):
		return self.inputdeck

inputTypes = {"a4db" : a4db_input, "binout" : Binout_input, "npz" : npz_input, "pkl" : pkl_input, "pickle": pkl_input, "py" : py_input, "k" : k_input, "key" : k_input}

# if __name__ == '__main__':
	
	# main_data = Input("./input_data/SFS_MAIN_Full_Model.a4db", "Stuff", False, False)
	# main_data = Input("./input_data/SFS_MAIN_SIMPMOD.a4db", None, False, False)

	# coordinates_data, displacement_data, full_data_ids, time_data = data_extraction_a4db("./input_data/SFS_MAIN_Full_Model.a4db", "Stuff", False, False)

	# data_stuff = main_data.extract_main()
	# print(data_stuff[1])
	# print(coordinates_data)
	# print(data_stuff)
	# print(np.array_equal(time_data, data_stuff[3]))

