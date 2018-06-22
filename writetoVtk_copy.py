from shutil import copyfile
import os
import os.path
import numpy as np

def writetoVtk(A, total_nodes, snapshot_selection, vtk_in_directory, vtk_out_directory, vtk_name, cellPerRow, isStaticNodes):

	# ensure output folder is created
	if not os.path.exists(vtk_out_directory):
		os.makedirs(vtk_out_directory)

	updatedVariableName = "POINTS"
	referenceVariableStart = "Deflection"
	referenceVariableEnd = "Velocity"
	metaDataStart = "CELLS"
	metaDataEnd = "CELL_TYPES"

	numStaticNodes = 4
	dimensions = 3
	# copy 0th vtk file to output folder
	# print("Working on problem 0")
	first_vtk_in = vtk_in_directory + vtk_name + '_0.vtk'
	first_vtk_out = vtk_out_directory + vtk_name + '_recon_0.vtk'
	copyfile(first_vtk_in, first_vtk_out)

	with open(first_vtk_in, 'r', encoding='utf-8') as vtk_in:
		bodyText = vtk_in.read()
		rv_start = bodyText.index(referenceVariableStart)
		rv_end = bodyText.index(referenceVariableEnd)
		md_start = bodyText.index(metaDataStart)
		md_end = bodyText.index(metaDataEnd)			

		metaData = [x[2:].strip().split(" ") for x in bodyText[md_start + (bodyText[md_start:].index('\n')) +1:md_end].strip().split("\n")]
		metaData = [int(item) for sublist in metaData for item in sublist]

		if(isStaticNodes):
			staticNodes = np.array(list(map(float, bodyText[rv_start:rv_end].split(" ")[-(numStaticNodes*dimensions+1):-1])))

	
	# loop through all timesteps
	for snapshot in snapshot_selection:
		x = np.concatenate((A[:,snapshot], staticNodes), axis=0)
		problem_in = vtk_name + '_' + str(snapshot+1)
		problem_out = vtk_name + '_recon_' + str(snapshot+1)
		vtk_in_name = vtk_in_directory + problem_in + '.vtk'
		vtk_out_name = vtk_out_directory + problem_out + '.vtk'		

		# identify location of variable that is to be updated in vtk file and isolate section of text that corresponds to it
		# the updated section is between pre_section_index : post_section_index
		with open(vtk_in_name, 'r', encoding='utf-8') as vtk_in:
			bodyText 	= 	vtk_in.read()
			uv_start 	= 	bodyText.index(updatedVariableName)
			uv_end 		= 	bodyText.index(metaDataStart)
			header 		= 	bodyText[uv_start:uv_start + (bodyText[uv_start:].index('\n')) +1]
			node_count 	= 	int(header.split(" ")[1])
			# dimensions = int(header.split(" ")[1])

			# reconstruct the list of modified variables from reconstructed A matrix
			counter = 0
			updatedVariables = ""
			for node in metaData:			
				updatedVariables = updatedVariables+" " + str(x[node])
				counter += 1
				if((counter)%cellPerRow == 0):
					updatedVariables = updatedVariables + "\n"

			updatedVariables = updatedVariables + "\n"
			# # write to output file at output location
			with open(vtk_out_name, 'w', encoding='utf-8') as vtk_out:
				pre_section = bodyText[:uv_start] + "\n"
				post_section = bodyText[uv_end:]
				vtk_out.write(pre_section + header + updatedVariables + post_section)


