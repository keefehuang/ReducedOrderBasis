from Binout_reading import binout_reading
from ristretto.mf import rsvd
from mask import randomMask
from mask import readNodes
from reduced_order import matrixReconstruction
from reduced_order import matrixReconstructionWithVelocity
from writetoVtk import writetoVtk
import numpy as np 
import scipy as sci
import small_func
import glob
import os.path
import re
from mapping import vtk_mapping
from mapping import node_mapping


if __name__ == '__main__':
	# Locate main directory folder to increase portability
	main_directory_path					=		os.path.dirname(os.path.realpath(__file__))

	#Import lsdyna input file from its own folder: LOOK FOR DYNA INPUT DECK FILES
	relative_data_path					=		'Data/'  #relative path for dyna-input-files 
	mask_path							=		main_directory_path
	input_file_name						=		os.path.join(main_directory_path, relative_data_path)

	dyna_input_files_available			=		[os.path.basename(x) for x in glob.glob('%s*.binout'%input_file_name)]
	k_input_files_available				=		[os.path.basename(x) for x in glob.glob('%s*.k'%input_file_name)]

	print('List of dyna input files available in /Data folder:')
	i = 0
	for name in dyna_input_files_available:
	    print('{first} = {second}'.format(first=i, second=name))
	    i+=1
	    
	choose_dyna_input_file =  input('choose dyna input file index = ')
	dyna_input_path        =  os.path.join(main_directory_path, relative_data_path, dyna_input_files_available[int(choose_dyna_input_file)])
	dyna_input_name        =  dyna_input_files_available[int(choose_dyna_input_file)]

	# location of the vtk in system
	vtk_in_folder = 'Visualization/VTK_IN/'
	vtk_out_folder = 'Visualization/VTK_OUT/'
	vtk_name = 'Bumper'
	vtk_in_directory = os.path.join(main_directory_path, vtk_in_folder) + vtk_name + '/'
	vtk_out_directory = os.path.join(main_directory_path, vtk_out_folder)  + vtk_name + '/'
	  

	# TODO: Add some form of options when running program.

	# Displacements, Coordinates, Velocities = binout_reading(dyna_input_name, True, 'coordinates + displacements + velocities')
	Coordinates, Displacements = binout_reading(dyna_input_name, False, 'coordinates + displacements')

	Movement = np.zeros((Displacements.shape[0],))
	for snapshot in range(1,Displacements.shape[1]):
		Movement += np.abs(Coordinates[:,snapshot] - Coordinates[:, snapshot-1])

	print(Displacements.shape)
	print(Coordinates.shape)
	stationary = np.where(Movement < 1e-12)
	print(stationary[0].shape)