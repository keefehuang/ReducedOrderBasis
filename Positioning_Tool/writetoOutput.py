# Add files in the parent folder to the path
import sys
import os
from os.path import splitext
import scipy
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from small_func import *
from Binout_reading import binout_reading
from mapping import *
from reduced_order import *
import fbpca
import importlib
from preprocessing import *
import pickle
import re
import sys 

def writetoOutput(keyfile, output_name, A_r, node_ids):
	node_start = "*NODE"
	
	node_ids   = iter(node_ids)
	A_r        = iter(A_r[:,-1])
	with open(keyfile, 'r') as file_in:
		bodyText = file_in.read()
		start_nodes = bodyText.index(node_start)
		end_nodes = re.search("\*[A-Z]+", bodyText[start_nodes+1:]).start() + start_nodes
		# raise NameError
		# end_nodes = bodyText.index(node_end)
		bodyText_start = bodyText[:start_nodes]
		bodyText_end = bodyText[end_nodes:]
		node_data = bodyText[start_nodes:end_nodes]
		node_line_data = node_data.split("\n")
		output_node_data =  []
		for line in node_line_data:
			# print(line)
			if(re.match(" [0-9. -]",line)):
				data = str(next(node_ids)).rjust(8) + str(round(next(A_r),3)).rjust(16) + str(round(next(A_r),3)).rjust(16) + str(round(next(A_r),3)).rjust(16)
				output_node_data.append(data)
			else:
				output_node_data.append(line)
				
		bodyTextNew = bodyText_start + "\n".join(output_node_data) + bodyText_end
 
	with open(output_name, 'w') as file_out:
		file_out.write(bodyTextNew)	
	

if __name__ == '__main__':
	main_directory_path						=		os.path.dirname(os.path.realpath(__file__))
	project_folder							= 		"Projects/"
	data_folder								=		"Data/"
	visualization_in_folder					=		"Visualization/VTK_IN/"
	visualization_out_folder				=		"Visualization/VTK_OUT/"
	mapping_folder 							=		"Visualization/Mapping/"
	project_files_available					= 		[x for x in next(os.walk("./Projects"))[1]]
	print("Please select project")
	for i, name in enumerate(project_files_available):
		print('{first} = {second}'.format(first=i, second=name))

	project_name 							= 		input('choose project folder = ')
	project_folder_path 					=		os.path.join(main_directory_path, project_folder, project_files_available[int(project_name)])
	data_folder_path						= 		os.path.join(project_folder_path, data_folder)

	supported_output_types 					=		[".k"]
	
	print("Please choose type of output file = ")
	
	for i, output_type in enumerate(supported_output_types):
		print('{first} = {second}'.format(first=i, second=output_type))

	output_type							=		input('choose output type = ')

	available_output_files					=		[os.path.basename(x) for x in glob.glob('%s*%s'%(data_folder_path, supported_output_types[int(output_type)]))]

	print('List of supported full data output files available in /Data folder:')
	for i, input_file in enumerate(available_output_files):
		print('{first} = {second}'.format(first=i, second=input_file))


	choose_output_file 					=  		input('choose output file index = ')
	output_name						=		available_output_files[int(choose_output_file)]
	output_file_path					=  		os.path.join(data_folder_path, output_name)

	writetoOutput(output_file_path)