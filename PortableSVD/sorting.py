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


def import_simplified_nodes(simplified_data_name, data_types):
	simplified_data_struct = np.load(simplified_data_name)
	data = []
	for data_name in data_types:
		data.append(simplified_data_struct[data_name])

	return data

### 
def sorting_nodes(ls_dyna_name, abaqus_name):
	start_name = '*NODE'
	end_name = "**\n** MEMBRANE and STRUCTURAL ELEMENTS"

	with open(abaqus_name, 'r', encoding='utf-8') as abaqus_in:
		bodyText = abaqus_in.read()
		# positions before and after node data
		in_start 			= 		bodyText.index(start_name)
		in_end 				= 		bodyText.index(end_name)
		
		re_coord = re.compile("[0-9,. -]{70,90}\n")
		match = re.findall(re_coord, bodyText[in_start:in_end])


	abaqus_ids = []
	abaqus_coordinates = []
	for coord in match:
		abaqus_ids.append(int(re.search("[0-9]*,", coord)[0][:-1]))
		abaqus_coordinates.append(list(map(float, re.search(",[0-9,. -]+\n", coord)[0][1:].strip().split(","))))

	abaqus_coordinates = np.array(abaqus_coordinates)

	ls_dyna_data = binout_reading(ls_dyna_name, False, 'coordinates')
	ls_dyna_ids = binout_reading(ls_dyna_name, False, 'ids')
	ls_dyna_coordinates		= small_func.rearange_xyz(ls_dyna_data)[:,0].reshape((-1,3))

	ind_lsdyna = np.argsort(ls_dyna_ids)
	ind_abaqus = np.argsort(abaqus_ids)

	sorted_lsdyna = ls_dyna_coordinates[ind_lsdyna]
	sorted_abaqus = abaqus_coordinates[ind_abaqus]
	error = 1e-2

	for i,node in enumerate(sorted_abaqus):
		count = 0
		error = 3
		switch = True
		while np.linalg.norm(sorted_abaqus[i] - sorted_lsdyna[i]) > error:
			count += 1

		if(count != 0 and switch):
			tmp = np.copy(sorted_lsdyna[i])
			sorted_lsdyna[i] = sorted_lsdyna[i+count]
			sorted_lsdyna[i+count] = tmp

			tmp =  np.copy(ind_lsdyna[i])
			ind_lsdyna[i] = ind_lsdyna[i+count]
			ind_lsdyna[i+count] = tmp
	index_map = ind_lsdyna[np.argsort(ind_abaqus)]

	no_vtk_nodes = sorted_abaqus.shape[0]
	extended_map = [0]*no_vtk_nodes*3

	for i,node in enumerate(index_map):
		extended_map[3*i] = node*3
		extended_map[3*i+1] = node*3 + 1
		extended_map[3*i+2] = node*3 + 2	

	np.save('abaqus_to_lsdyna' + "bumper", extended_map)	

	return 'extended_map' + "bumper"

	# for num in range(1,22548):
	# 	if np.linalg.norm(abaqus_coordinates[num-1] - ls_dyna_coordinates[num-1]) > error:
	# 		print(np.linalg.norm(abaqus_coordinates[num-1] - ls_dyna_coordinates[num-1]))
	# 		print(abaqus_coordinates[num-1])
	# 		print(ls_dyna_coordinates[num-1])
	# 		print("Error at", num-1)
	# 		break



if __name__ == '__main__':
	abaqus_name = "Data/WS_neu_mitMPCundMasse_v9_500kg__Traegheit_3Contact.inp"
	ls_dyna_name = "bumper.binout"

	sorting_nodes(ls_dyna_name, abaqus_name)