# from scipy import ndimage 
import numpy as np 
import scipy as sci
import small_func
import glob
import os.path
import re

def input_data(Data, point, locationOfPoints, bodyText):
	for direction in range(3):
		pointData = []
		matchObject = next(locationOfPoints)
		matchSpan = bodyText[matchObject.span()[0]:matchObject.span()[1]].split("\n")
		for matchAttribute in matchSpan[3:-1]:
			pointData.append(float(re.split(" +",matchAttribute.strip())[1]))
		if point == 0 and direction == 0:
			Data = np.array(pointData).reshape((1,-1))
		else:
			pointData = np.array(pointData).reshape((1,-1))
			Data = np.concatenate((Data,pointData), axis=0)
	return Data, matchSpan


def read_simplified_data(file_name, problem_name):
	# points for bumper is 41
	# points = int(input("Number of nodes:\n"))
	# Timesteps for bumper is 202
	# timesteps = int(input("Number of timesteps:\n"))
	# Type of simplified data
	

	with open(file_name, 'r', encoding='utf-8') as WS:
		bodyText = WS.read()
		coords = re.compile('Title.{15,30}\[Node_PART-1-[0-9\.]+\]?\nAbs_unit TI\nOrd_unit ..\n( +[0-9\.\+e\-]+\n*){202,2500}', re.S)
		header = re.compile('Title.{15,30}\[Node_PART-1-[0-9\.]+\]?\nAbs_unit TI\nOrd_unit ..\n')
		
		locationOfPoints = re.finditer(coords, bodyText)
		count = 0
		for item in locationOfPoints:
			count += 1
			print(item[0][:50])
		headerPoints = re.findall(header, bodyText)
		points = int(count/7/3)

		Coordinates = np.array([])
		Displacements = np.array([])
		AngularDisplacements = np.array([])
		Velocities = np.array([])
		AngularVelocities = np.array([])
		Accelerations = np.array([])
		AngularAccelerations = np.array([])

		locationOfPoints = re.finditer(coords, bodyText)
		for point in range(points):
			print(point)
			Accelerations, matchSpan 			= 	input_data(Accelerations, point, locationOfPoints, bodyText)
			AngularAccelerations, matchSpan 	= 	input_data(AngularAccelerations, point, locationOfPoints, bodyText)
			Coordinates, matchSpan 				= 	input_data(Coordinates, point, locationOfPoints, bodyText)
			Displacements, matchSpan 			= 	input_data(Displacements, point, locationOfPoints, bodyText)
			AngularDisplacements, matchSpan 	= 	input_data(AngularDisplacements, point, locationOfPoints, bodyText)
			Velocities, matchSpan			 	= 	input_data(Velocities, point, locationOfPoints, bodyText)
			AngularVelocities, matchSpan 		= 	input_data(AngularVelocities, point, locationOfPoints, bodyText)
			
		time = []
		for matchAttribute in matchSpan[3:-1]:
			time.append(float(re.split(" +",matchAttribute.strip())[0]))

		Time = np.array(time)
		print(Time)
			
	np.savez(problem_name + "_data", Coordinates=Coordinates, Displacements=Displacements, AngularDisplacements=AngularDisplacements, Velocities=Velocities, AngularVelocities=AngularVelocities, Accelerations=Accelerations, AngularAccelerations=AngularAccelerations, Time = Time)

	return problem_name+"_data.npz"

# def compareSimplifiedNodes():
# 	simplified_data_1 = np.load("bumper_data.npz")	
# 	simplified_data_2 = np.load("Bumper_data.npz")	

# 	Coords1 = simplified_data_1["Coordinates"][:,0].reshape((-1,3))
# 	Coords2 = simplified_data_2["Coordinates"][:,0].reshape((-1,3))

# 	for item_1 in Coords1:
# 		for item_1 in Coords2:
			
# This program is mainly to read abaqus output and store it in readable arrays similar to that from Binout data.
if __name__ == '__main__':
	# main_directory_path					=		os.path.dirname(os.path.realpath(__file__))
	file_name		=		'/home/keefe/Documents/BMW/HiWi/Code/ReducedOrderBasis/Data/Bumper/Higher_Resolution/SFS_MAIN_NO_ROT.crv'
	# input_file_name						=		os.path.join(main_directory_path, relative_simplified_data_path)
	problem_name 						= 		"bumper"

	read_simplified_data(file_name, problem_name)
	# compareSimplifiedNodes();