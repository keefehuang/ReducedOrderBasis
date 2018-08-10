from Binout_reading import binout_reading
from ristretto.mf import rsvd
from mask import randomMask
from mask import readNodes
import os.path
import glob
import numpy as np 	
import small_func
import re


if __name__ == '__main__':

	# # Locate main directory folder to increase portability
	# main_directory_path					=		os.path.dirname(os.path.realpath(__file__))

	# #Import lsdyna input file from its own folder: LOOK FOR DYNA INPUT DECK FILES
	# relative_data_path					=		'Data/'  #relative path for dyna-input-files 
	# relative_simplified_data_path		=		'Data/Bumper/' #relative path for data from simplified model
	# mask_path							=		main_directory_path
	# input_file_name						=		os.path.join(main_directory_path, relative_data_path)
	# data_input_file_name				=		os.path.join(main_directory_path, relative_simplified_data_path)

	# dyna_input_files_available			=		[os.path.basename(x) for x in glob.glob('%s*.binout'%input_file_name)] #dyna files available in provided data folder
	# k_input_files_available				=		[os.path.basename(x) for x in glob.glob('%s*.k'%input_file_name)] # input decks available


	# print('List of dyna input files available in ' + relative_data_path + ' folder:')
	# i = 0
	# for name in dyna_input_files_available:
	#     print('{first} = {second}'.format(first=i, second=name))
	#     i+=1
	    
	# choose_dyna_input_file =  input('choose dyna input file index = ')
	# dyna_input_path        =  os.path.join(main_directory_path, relative_data_path, dyna_input_files_available[int(choose_dyna_input_file)])
	# dyna_input_name        =  dyna_input_files_available[int(choose_dyna_input_file)]

	# simplified_data_name = 'Bumper_data.npz'


	# # reads time, coordinates and displacements from Binout (ie. full model)
	# binout_coordinates, binout_displacements, binout_angular_velocities = binout_reading(dyna_input_name, False, 'coordinates + displacements + rvelocities')
	# binout_timesteps = binout_reading(dyna_input_name, False, 'time')

	# # small_func rearranges the output matrix A to follow an x-y-z arrangement along index 0
	# # TODO: Create function to rearrange binout per node order
	# binout_displacements 						= small_func.rearange_xyz(binout_displacements)
	# binout_coordinates 							= small_func.rearange_xyz(binout_coordinates)
	# binout_angular_velocities 					= small_func.rearange_xyz(binout_angular_velocities)

	# simplified_data = np.load(simplified_data_name)

	# node_data_location = '/home/keefe/Documents/BMW/HiWi/Code/ReducedOrderBasis/XSectionNodes/'
	# node_data = np.load(node_data_location + 'Node0.npy')
	# first_node = int(node_data[0]) - 1

	# binout_coordinates = binout_coordinates[:,0]

	# test_nodes = [3*first_node, 3*first_node+1, 3*first_node+2]

	# print(binout_coordinates[test_nodes])

	main_directory_path							=		os.path.dirname(os.path.realpath(__file__))
	relative_data_path							=		'XSectionNodes/'  #relative path for node xsection files 
	input_file_name								=		os.path.join(main_directory_path, relative_data_path)
	node_xsection_files_available				=		[os.path.basename(x) for x in glob.glob('%s*.npy'%input_file_name)] # input decks available

	nodes = []

	total_nodes = 41
	for num in range(total_nodes):
		try:
			node = np.load(input_file_name + 'Node' + str(num) + '.npy')
			nodes.append(node)
		except:
			nodes.append([])

	np.save('XSectionNodes', nodes)



	# for i, file_name in enumerate(node_xsection_files_available):

