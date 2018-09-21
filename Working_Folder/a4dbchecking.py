import h5py
import numpy as np 
from Binout_reading import *
from small_func import *

# 6 - x


def main():
	a4db_name 		= 	"./input_data/SFS_MAIN_Full_Model.a4db"
	a4db 			= 	h5py.File(a4db_name, "r")

	velocityx 		= 	np.array(a4db["model_0"]["results_0"]["step_20"]["function_6"]["node"]["function"])
	velocityy 		= 	np.array(a4db["model_0"]["results_0"]["step_71"]["function_10"]["node"]["function"])
	velocityz 		= 	np.array(a4db["model_0"]["results_0"]["step_20"]["function_8"]["node"]["function"])


	print(velocity.shape)
	# print(velocity)


	# print(np.max(velocityx))
	# print(np.min(velocityx))
	# print(np.average(velocityx))
	# print("\n")

	print(velocityy.shape)
	print(np.max(velocityy))
	print(np.min(velocityy))
	print(np.average(velocityy))
	print("\n")

	# print(np.max(velocityz))
	# print(np.min(velocityz))
	# print(np.average(velocityz))
	# print("\n")

	binout_name		= 	"./input_data/bumper.binout"
	velocity_data 	= rearange_xyz(binout_reading(binout_name, False, 'velocities'))[:,70].reshape((-1,3))


	# print(velocity_data.shape)
	
	# print(np.max(velocity_data[:,0]))
	# print(np.min(velocity_data[:,0]))
	# print(np.average(velocity_data[:,0]))
	# print("\n")

	print(np.max(velocity_data[:,1]))
	print(np.min(velocity_data[:,1]))
	print(np.average(velocity_data[:,1]))
	print("\n")

	# print(np.max(velocity_data[:,2]))
	# print(np.min(velocity_data[:,2]))
	# print(np.average(velocity_data[:,2]))

if __name__ == '__main__':
	main()