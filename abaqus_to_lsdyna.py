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




if __name__ == '__main__':
	### Locate main directory folder to increase portability
	main_directory_path					=		os.path.dirname(os.path.realpath(__file__))

	### Import lsdyna input file from its own folder: LOOK FOR DYNA INPUT DECK FILES
	relative_data_path					=		'Data/'  #relative path for dyna-input-files 
	relative_simplified_data_path		=		'Data/Bumper/' #relative path for data from simplified model
	relative_node_xsection_path			=		'XSectionNodes/Original/'  #relative path for node xsection files 
	
	mask_path							=		main_directory_path
	input_file_name						=		os.path.join(main_directory_path, relative_data_path)
	data_input_file_name				=		os.path.join(main_directory_path, relative_simplified_data_path)
	node_xsection_file_name				= 		os.path.join(main_directory_path, relative_node_xsection_path)

	dyna_input_files_available			=		[os.path.basename(x) for x in glob.glob('%s*.binout'%input_file_name)] #dyna files available in provided data folder
	
	choose_dyna_input_file =  input('choose dyna input file index = ')
	dyna_input_path        =  os.path.join(main_directory_path, relative_data_path, dyna_input_files_available[int(choose_dyna_input_file)])
	dyna_input_name        =  dyna_input_files_available[int(choose_dyna_input_file)]

	binout_coordinates 		= binout_reading(dyna_input_name, False, 'coordinates')
	binout_ind 				= np.lexsort(binout_reading(dyna_input_name, False, 'ids'))
	binout_coordinates 		= binout_coordinates[binout_ind]


	abaqus_data 			= np.load("Full_Node_Data_High_Resolution.npy")
	abaqus_ind 				= np.lexsort((simplified_data[:,0]))
	abaqus_data 			= abaqus_data[abaqus_ind]


	
	print(binout_coordinates)

	print(abaqus_data	)



	for i,node in enumerate(simplified_data_arranged):
		if node - binout