import numpy as np
import os.path
import re


# Generates a random set of n nodes.
def randomMask(total_nodes, n):
	if n > total_nodes:
		print('Cannot sample more values than existing nodes')
		return

	c = range(total_nodes)

	initMask = np.random.choice(c, n, replace=False)*3
	mask = np.append(initMask, initMask+1)
	mask = np.append(mask, initMask+2)

	return np.sort(np.array(mask))

# Takes in a set of nodes selected from the LS-DYNA model. Maps the selected nodes to the rows in the Binout data using Node IDs
# in the ls-dyna .key file
def readNodes(node_selection, total_nodes, k_input_name=None):
	node_selection = set(node_selection)
	if max(node_selection) >= total_nodes or min(node_selection) < 0:
		raise ValueError("Chosen nodes not valid, either chosen nodeID too large or negative")
		return

	header 	= "*NODE"
	end 	= "*ELEMENT_SOLID"
	mapping = []
	mapped_nodes = []
	if(k_input_name):
		with open(k_input_name, 'r', encoding='utf-8') as k_input:
			bodyText = k_input.read()
			node_start 			= 		bodyText.index(header)
			node_end 			= 		bodyText.index(end)
			node_list			=		bodyText[node_start+ (bodyText[node_start:].index('\n')):node_end].split('\n')

		for line in node_list[1:-1]:
			mapping.append(int((re.match("[0-9]*", line.strip())[0])))

		for node in node_selection:
			mapped_nodes.append(mapping.index(node))

	mapped_nodes = np.array([int(x) for x in mapped_nodes])*3

	temp = np.append(mapped_nodes, mapped_nodes+1)
	temp = np.append(temp, mapped_nodes+2)

	return np.sort(np.array(temp))

