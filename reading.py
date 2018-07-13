# from scipy import ndimage 
from mask import randomMask
from reduced_order import matrixReconstruction
from reduced_order import matrixReconstructionWithVelocity
from writetoVtk import writetoVtk
import numpy as np 
import scipy as sci
import small_func
import glob
import os.path
import re


def read_simplified_data(data_location, problem_name):
	# points for bumper is 41
	points = int(input("Number of nodes:\n"))
	# Timesteps for bumper is 202
	timesteps = int(input("Number of timesteps:\n"))
	# Type of simplified data
	data_type = input("What kind of input file are you reading?\n")

	data_type = '%s*' + data_type

	print(data_type	)

	input_files_available			=		[os.path.basename(x) for x in glob.glob(data_type%data_location)]

	print('List of input files available in folder:')
	i = 0
	for name in input_files_available:
	    print('{first} = {second}'.format(first=i, second=name))
	    i+=1

	choose_input_file =  input('choose input file index = ')
	input_path        =  os.path.join(data_location, input_files_available[int(choose_input_file)])

	with open(input_path, 'r', encoding='utf-8') as WS:
		bodyText = WS.read()
		coords = re.compile('Title.{9,14}\[Node_PART-1-[0-9\.]+\]?\nAbs_unit TI\nOrd_unit ..\n( +[0-9\.\+e\-]+\n*){202}', re.S)
		header = re.compile('Title.{9,14}\[Node_PART-1-[0-9\.]+\]?\nAbs_unit TI\nOrd_unit ..\n')
		
		locationOfPoints = re.finditer(coords, bodyText)
		headerPoints = re.findall(header, bodyText)

		Coordinates = np.array([])
		Displacements = np.array([])
		AngularRotations = np.array([])

		for point in range(points):
			for direction in range(3):
				pointCoordinates = []
				matchObject = next(locationOfPoints)
				matchSpan = bodyText[matchObject.span()[0]:matchObject.span()[1]].split("\n")
				for matchAttribute in matchSpan[3:-1]:
					pointCoordinates.append(float(re.split(" +",matchAttribute.strip())[1]))
				if point == 0 and direction == 0:
					Coordinates = np.array(pointCoordinates).reshape((1,101))
				else:
					pointCoordinates = np.array(pointCoordinates).reshape((1,101))
					Coordinates = np.concatenate((Coordinates,pointCoordinates), axis=0)
			for direction in range(3):
				pointDisplacements = []
				matchObject = next(locationOfPoints)
				matchSpan = bodyText[matchObject.span()[0]:matchObject.span()[1]].split("\n")
				for matchAttribute in matchSpan[3:-1]:
					pointDisplacements.append(float(re.split(" +",matchAttribute.strip())[1]))
				if point == 0 and direction == 0:
					Displacements = np.array(pointDisplacements).reshape((1,101))
				else:
					pointDisplacements = np.array(pointDisplacements).reshape((1,101))
					Displacements = np.concatenate((Displacements,pointDisplacements), axis=0)
			for direction in range(3):
				pointAngularRotations = []
				matchObject = next(locationOfPoints)
				matchSpan = bodyText[matchObject.span()[0]:matchObject.span()[1]].split("\n")
				for matchAttribute in matchSpan[3:-1]:
					pointAngularRotations.append(float(re.split(" +",matchAttribute.strip())[1]))
				if point == 0 and direction == 0:
					AngularRotations = np.array(pointAngularRotations).reshape((1,101))
				else:
					pointAngularRotations = np.array(pointAngularRotations).reshape((1,101))
					AngularRotations = np.concatenate((AngularRotations,pointAngularRotations), axis=0)

		time = []
		for matchAttribute in matchSpan[3:-1]:
			time.append(float(re.split(" +",matchAttribute.strip())[0]))

		Time = np.array(time)
			
	np.savez(problem_name + "_data", Coordinates=Coordinates, Displacements=Displacements, AngularRotations=AngularRotations, Time = Time)

	return problem_name+"_data.npz"

# This program is mainly to read abaqus output and store it in readable arrays similar to that from Binout data.
if __name__ == '__main__':
	main_directory_path					=		os.path.dirname(os.path.realpath(__file__))
	relative_simplified_data_path		=		'Data/Bumper/'
	input_file_name						=		os.path.join(main_directory_path, relative_simplified_data_path)
	read_simplified_data(input_file_name)
