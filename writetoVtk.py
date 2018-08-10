from shutil import copyfile
import os
import os.path
import numpy as np


def writetoVtk(A_r, total_nodes, snapshot_selection, vtk_in_directory, vtk_out_directory, vtk_name, cellPerRow, isStaticNodes, mapping_name, isError=False, error_r=None):

	# ensure output folder is created
	if not os.path.exists(vtk_out_directory):
		os.makedirs(vtk_out_directory)

	updatedVariableName = "POINTS"
	referenceVariableStart = "Deflection"
	referenceVariableEnd = "Velocity"
	metaDataStart = "CELLS"
	index_map = np.load(mapping_name)
	vtk_nodes = index_map.shape[0]
	if isStaticNodes:
		numStaticNodes = 4
	else:
		numStaticNodes = 0
	dimensions = 3
	# determines location of reference
	ref_vtk_in = vtk_in_directory + vtk_name + '_1.vtk'

	with open(ref_vtk_in, 'r', encoding='utf-8') as vtk_in:
		bodyText = vtk_in.read()
		uv_start          =       bodyText.index(updatedVariableName)
		uv_end              =       bodyText.index(metaDataStart)
		rv_start            =       bodyText.index(referenceVariableStart)
		rv_end              =       bodyText.index(referenceVariableEnd)
		uv_header           =       bodyText[uv_start:uv_start + (bodyText[uv_start:].index('\n')) +1] + "\n"
		rv_header           =       bodyText[rv_start:rv_start + (bodyText[rv_start:].index('\n')) +1] + "\n"
		node_count          =       int(uv_header.split(" ")[1])

		pre_section = bodyText[:uv_start] + "\n"
		mid_section = bodyText[uv_end:rv_start] + "\n"
		post_section = bodyText[rv_end:]
		if isStaticNodes:
			staticNodesAppend = " ".join(bodyText[rv_start:rv_end].split(" ")[-(numStaticNodes*dimensions+1):-1])

	# loop through all timesteps
	for snapshot in snapshot_selection:
		x = A_r[:,snapshot]
		# problem_in = vtk_name + '_' + str(snapshot)
		problem_out = vtk_name + '_recon_' + str(snapshot)
		print("Printing problem {}".format(snapshot))
		# vtk_in_name = vtk_in_directory + problem_in + '.vtk'
		vtk_out_name = vtk_out_directory + problem_out + '.vtk'		

		x = x[index_map]
		reshaped_x = x[:vtk_nodes-vtk_nodes%9].reshape(((vtk_nodes-vtk_nodes%9)//9,9))
		updatedVariables = "\n".join([" ".join(map(str,line)) for line in reshaped_x]) + "\n"

		for variable in x[vtk_nodes-vtk_nodes%9:]:
			updatedVariables = updatedVariables + " " + str(variable)

		updatedVariables = updatedVariables + "\n"

		if isError:
			# errorHeader = "POINT_DATA " + str(int(len(index_map)/3 + numStaticNodes)) + "\n\nVECTORS Error float " + str(int(len(index_map)/3 + numStaticNodes)) + "\nLOOKUP_TABLE default \n"
			errorHeader = "\n FIELD FieldData 1 \n\n Error " + str(dimensions) + " " + str(int(len(index_map)/3 + numStaticNodes)) + " float " +  "\n"
			errorVariable = ""
			error_x = error_r[: , snapshot]
			error_x = error_x[index_map]

			reshaped_error = error_x[:vtk_nodes-vtk_nodes%9].reshape(((vtk_nodes-vtk_nodes%9)//9,9))
			errorVariable = "\n".join([" ".join(map(str,line)) for line in reshaped_error]) + "\n"

			for variable in error_x[vtk_nodes-vtk_nodes%9:]:
				errorVariable = errorVariable + " " + str(variable)

			if isStaticNodes:
				for i in range(len(index_map), len(index_map)+numStaticNodes*dimensions):
					errorVariable = errorVariable + " 0"
					if (i+1)%(cellPerRow*dimensions) == 0:
						errorVariable = errorVariable + "\n"

		if isStaticNodes:
			updatedVariables = updatedVariables + " " + staticNodesAppend + "\n"
		
		# # write to output file at output location
		with open(vtk_out_name, 'w', encoding='utf-8') as vtk_out:
			if isError:
				# vtk_out.write(pre_section + uv_header + updatedVariables + errorHeader + errorVariable + mid_section + rv_header + updatedVariables + post_section)	
				vtk_out.write(pre_section + uv_header + updatedVariables + mid_section + rv_header + updatedVariables + post_section + errorHeader + errorVariable)	
			else:
				vtk_out.write(pre_section + uv_header + updatedVariables + mid_section + rv_header + updatedVariables + post_section)


