import numpy as np 



if __name__ == '__main__':
	node_num = input("Input node num:\n")
	simplified_node_name = 'Node' + str(node_num) + '.npy'
	simplified_data = np.load(simplified_node_name)

	# simplified_coordinates 		= 	simplified_data['Coordinates']

	# num_simplified_nodes = simplified_coordinates.shape[0]//3
	for i, vertex in enumerate(simplified_data):
		print("Vertex {}: {}".format(i, vertex))

	check = 0
	while check != 1:
		check = int(input("Is the check correct?\n"))

		if check == 1:
			pass
		else:
			vertex = int(input("Which vertex to update?\n"))
			simplified_data[vertex] = input('Store vertex {} in node {}\n'.format(vertex, node_num))

	np.save('Node' + str(node_num), simplified_data)		
	# snapshot = 0
	# simplified_nodes = np.reshape(simplified_coordinates[:,0], (num_simplified_nodes,3))

	# print(simplified_nodes)