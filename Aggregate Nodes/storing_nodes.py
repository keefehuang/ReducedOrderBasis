import numpy as np 

nodes = 41
vertexes = 68

if __name__ == '__main__':
	node_number = input("Which node are you inputting?\n")
	node_store = np.zeros((68))

	for vertex in range(vertexes):
		while node_store[vertex] == 0:
			try:
				node_store[vertex] = int(input('Store vertex {} in node {}\n'.format(vertex, node_number)))
			except ValueError:
				print("Invalid input value\n")
				node_store[vertex] = 0

	np.save('Node' + str(node_number), node_store)

	check = 0
	while check != 1:
		check = int(input("Is the check correct?\n"))

		if check == 1:
			pass
		else:
			vertex = int(input("Which vertex to update?\n"))
			node_store[vertex] = input('Store vertex {} in node {}\n'.format(vertex, node_number))

	np.save('Node' + str(node_number), node_store)