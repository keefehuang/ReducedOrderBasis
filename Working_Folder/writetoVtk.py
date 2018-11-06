from shutil import copyfile
import os
import os.path
import numpy as np
try:
	import pyprind
except:
	pass


def writetoVtk(A_r, full_node_num, input_vtk_file, output_vtk_file, mapping_name, isCalculateError=False, error_r=None):
	snapshot_selection = range(A_r.shape[1])
	output_folder = "/".join(output_vtk_file.split("/")[:-1])
	# ensure output folder is created
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	updatedVariableName = "POINTS"
	referenceVariableStart = "Deflection"
	referenceVariableEnd = "Velocity"
	metaDataStart = "METADATA\nINFORMATION 2"
	index_map = np.load(mapping_name)
	vtk_nodes = index_map.shape[0]

	dimensions = 3

	with open(input_vtk_file, 'r') as vtk_in:
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

	# loop through all timesteps
	try:
		bar = pyprind.ProgBar(len(snapshot_selection), monitor=True, title='Printing vtks', bar_char='*')
	except:
		print("Printing vtks")
	for snapshot in snapshot_selection:
		x = A_r[:,snapshot]
		vtk_out_name = output_vtk_file[:-4] + str(snapshot) + '.vtk'		

		x = x[index_map]
		reshaped_x = x[:vtk_nodes-vtk_nodes%9].reshape(((vtk_nodes-vtk_nodes%9)//9,9))
		updatedVariables = "\n".join([" ".join(map(str,line)) for line in reshaped_x]) + "\n"

		for variable in x[vtk_nodes-vtk_nodes%9:]:
			updatedVariables = updatedVariables + " " + str(variable)

		updatedVariables = updatedVariables + "\n"

		if isCalculateError:
			# errorHeader = "POINT_DATA " + str(int(len(index_map)/3 + numStaticNodes)) + "\n\nVECTORS Error float " + str(int(len(index_map)/3 + numStaticNodes)) + "\nLOOKUP_TABLE default \n"
			errorHeader = "\n FIELD FieldData 1 \n\n Error " + str(dimensions) + " " + str(int(len(index_map)/3)) + " float " +  "\n"
			errorVariable = ""
			error_x = error_r[index_map , snapshot]

			reshaped_error = error_x[:vtk_nodes-vtk_nodes%9].reshape(((vtk_nodes-vtk_nodes%9)//9,9))
			errorVariable = "\n".join([" ".join(map(str,line)) for line in reshaped_error]) + "\n"

			for variable in error_x[vtk_nodes-vtk_nodes%9:]:
				errorVariable = errorVariable + " " + str(variable)

		##  write to output file at output location
		with open(vtk_out_name, 'w') as vtk_out:
			if isCalculateError:
				vtk_out.write(pre_section + uv_header + updatedVariables + mid_section + rv_header + updatedVariables + post_section + errorHeader + errorVariable)	
			else:
				vtk_out.write(pre_section + uv_header + updatedVariables + mid_section + rv_header + updatedVariables + post_section)
		try:
			bar.update()
		except:
			pass


