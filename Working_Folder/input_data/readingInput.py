from shutil import copyfile
import os
import os.path
import numpy as np
try:
	import pyprind
except:
	pass
import re
import vtkInterface
from qd.cae.dyna import KeyFile

def reada4db():
	with open("WS_neu_mitMPCundMasse_v9_500kg__Traegheit_3Contact.inp") as f:
		bodyText = f.read()
		# node_search = re.compile('*NODE\n( +[0-9\.\+e\-,]+\n*){202,2500}', re.S)
		node_search = re.compile('\*NODE\n( +[0-9\.\+e\-, ]+\n){202,25000}', re.S)
		cell_search = re.compile('\*ELEMENT[, A-Z=;0-9_]*\n( +[0-9\.\+e\-, ]+\n*){3,150000}', re.S)

		locationOfNodes = re.finditer(node_search, bodyText)
		locationOfCells = re.finditer(cell_search, bodyText)

		vertices = [list(map(float, item.split(",")[1:])) for item in next(locationOfNodes)[0].split("\n")][:-1]

		cellPoints = None
		cellIds	= None
		for items in locationOfCells:
			if cellPoints is None:
				cellPoints = [list(map(float, item.split(",")[1:])) for item in items[0].split("\n")[1:]][:-1]
				# print(len(cellPoints))
				
				cellPoints = [ [len(item)] + item for item in cellPoints]				
				cellIds = [item.split(",")[0] for item in items[0].split("\n")[1:]][:-1]
			else:
				
				newCellPoints = [list(map(float, item.split(",")[1:])) for item in items[0].split("\n")[1:]][:-1]
				newCellPoints = [ [len(item)] + item for item in newCellPoints]
				cellPoints = cellPoints + newCellPoints
				cellIds = cellIds + [item.split(",")[0] for item in items[0].split("\n")[1:]][:-1]
		
		cellPoints = np.hstack(cellPoints)
		cellIds = np.array(cellIds)
		vertices[0] = [0,0,0]
		vertices = np.array(vertices)

		
		# print(cellIds.shape)
		surf = vtkInterface.PolyData(vertices, cellPoints)

		# surf.Plot()
		surf.Write("sample.vtk")
		# cellsVertices = [item.split(",") for item in next(locationOfCells)[0].split("\n")[1:-1]]
		# for node in vertices:
		# 	print(node)

		# for cell in cellsVertices:
		# 	print(cell)
		# print(locationOfCells)

if __name__ == '__main__':
	keyFileBumper = KeyFile("SFS_Dyna_main.key", read_keywords=True, parse_mesh=True)
	keyFileBarrier = KeyFile("OKL_KOM_Barriere_Front.key", read_keywords=True)

	# for key in keyFileBumper.keys():
	# 	print(key)


	# for item in keyFileBumper.get_element_coords():
	# 	print(item)

	# for item in str(keyFileBumper.get_element_ids()).split("\n"):
	# 	print(item)		

	for item in keyFileBarrier.keys():
		print(item)

	for item in str(keyFileBarrier.get_element_ids()).split("\n"):
		print(item)		