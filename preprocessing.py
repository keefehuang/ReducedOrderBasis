from Binout_reading import binout_reading
from ristretto.mf import rsvd
# from scipy import ndimage 
from mask import randomMask
from mask import readNodes
from reduced_order import matrixReconstruction
from reduced_order import matrixReconstructionWithVelocity
from writetoVtk import writetoVtk
# import matplotlib.pyplot as plt
import numpy as np 
import scipy as sci
import small_func
import glob
import os.path
import re
from mapping import vtk_mapping
from mapping import node_mapping



def preprocessingInputs(dyna_input_name, selected_node_data):
	time_error = 1e-3

	# reads coordinates and displacements from Binout
	Coordinates, Displacements = binout_reading(dyna_input_name, False, 'coordinates + displacements')
	Times = binout_reading(dyna_input_name, False, 'time')

	# small_func rearranges the output matrix A to follow an x-y-z arrangement along index 0
	Displacements = small_func.rearange_xyz(Displacements)
	Coordinates = small_func.rearange_xyz(Coordinates)

	Movement = np.zeros((Displacements.shape[0],))
	for snapshot in range(1,Displacements.shape[1]):
		Movement += np.abs(Displacements[:,snapshot] - Displacements[:, snapshot-1])
	moving_nodes_indices = np.where(Movement > 1e-12)
	movingDisplacements = Displacements[moving_nodes_indices[0],:]

	selected_nodes, node_indices, selected_nodes_timing = node_mapping(movingDisplacements, selected_node_data)

	timeindices = []
	for timeindex in selected_nodes_timing:
		timeindices.append(np.where(Times - timeindex < time_error)[0][0])

	Displacements = Displacements[:, timeindices]
	Coordinates = Coordinates[:, timeindices]

	for i in range(0,Coordinates.shape[1]):
		if np.linalg.norm(Coordinates[:,i])!=0:
			Coordinates[:,i]=Coordinates[:,i]/np.linalg.norm(Coordinates[:,i])
			Displacements[:,i]=Displacements[:,i]/np.linalg.norm(Coordinates[:,i])
			nodes[:,i]=nodes[:,i]/np.linalg.norm(Coordinates[:,i])
	return selected_nodes, node_indices, Coordinates, Displacements, movingDisplacements, moving_nodes_indices