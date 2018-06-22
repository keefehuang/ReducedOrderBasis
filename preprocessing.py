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


def preprocessingInputs(dyna_input_name, selected_node_name):
	# # read time, coordinates and displacements from simplified model
	simplifiedData = np.load(selected_node_name)
	simplifiedTimes = simplifiedData['Time']
	simplifiedCoordinates = simplifiedData['Coordinates']
	simplifiedDisplacements = simplifiedData['Displacements']

	# reads time, coordinates and displacements from Binout
	binoutCoordinates, binoutDisplacements = binout_reading(dyna_input_name, False, 'coordinates + displacements')
	binoutTimes = binout_reading(dyna_input_name, False, 'time')

	# small_func rearranges the output matrix A to follow an x-y-z arrangement along index 0
	binoutDisplacements = small_func.rearange_xyz(binoutDisplacements)
	binoutCoordinates = small_func.rearange_xyz(binoutCoordinates)

	# removes the boundary condition nodes (ie nodes that do not move) Movement array stores the displacement over time for all nodes
	# if the displacement is > 1e-12 over the full snapshot, we consider that the node is part of the simulation. Otherwise, we remove 
	# it from consideration when mapping selected node data to Binout data
	Movement = np.zeros((binoutDisplacements.shape[0],))
	for snapshot in range(1,binoutDisplacements.shape[1]):
		Movement += np.abs(binoutDisplacements[:,snapshot] - binoutDisplacements[:, snapshot-1])

	# Moving displacements/coordinates indicate the nodes which move during the simulation
	binoutMovingIndices = np.where(Movement > 1e-12)[0]
	binoutMovingDisplacements = binoutDisplacements[binoutMovingIndices,:]
	binoutMovingCoordinates = binoutCoordinates[binoutMovingIndices,:]

	# randMask = randomMask(binoutMovingCoordinates.shape[0]//3, 41)
	# simplifiedCoordinates = binoutMovingCoordinates[randMask, :]
	# simplifiedDisplacements = binoutMovingDisplacements[randMask, :]

	# the timesteps of the data from the simplified simulation may not match the timesteps in the binout. Ensure that the
	# timesteps match
	time_error = 1e-5
	time_indices = []
	for timeindex in simplifiedTimes:
		time_indices.append(np.where(abs(binoutTimes - timeindex) < time_error)[0][0])

	binoutDisplacements = binoutDisplacements[:, timeindices]
	binoutCoordinates = binoutCoordinates[:, timeindices]
	binoutMovingDisplacements = binoutMovingDisplacements[:,timeindices]
	binoutMovingCoordinates = binoutMovingCoordinates[:,timeindices]

	simplifiedDisplacements, simplifiedNodeIndices = node_mapping(binoutMovingCoordinates, simplifiedCoordinates, simplifiedDisplacements)

	for i in range(0,binoutCoordinates.shape[1]):
		if np.linalg.norm(binoutCoordinates[:,i])!=0:
			binoutCoordinates[:,i]=binoutCoordinates[:,i]/np.linalg.norm(binoutCoordinates[:,i])
			binoutDisplacements[:,i]=binoutDisplacements[:,i]/np.linalg.norm(binoutCoordinates[:,i])
			simplifiedDisplacements[:,i]=simplifiedDisplacements[:,i]/np.linalg.norm(binoutCoordinates[:,i])
			binoutMovingDisplacements[:,i]=binoutMovingDisplacements[:,i]/np.linalg.norm(binoutCoordinates[:,i])

	return simplifiedDisplacements, simplifiedNodeIndices, binoutCoordinates, binoutDisplacements, binoutMovingDisplacements, binoutMovingIndices, time_indices