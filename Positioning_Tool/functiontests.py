from mapping import *
import numpy as np
import scipy
import unittest
import pickle


def isEqual(x, y, error):
	if(scipy.linalg.norm(x-y) < error):
		return True
		return False

class TestStringMethods(unittest.TestCase):


	def test_append_rows(self):

		example_matrix = [[1, 1, 1, 1, 1],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0], [2, 2, 2, 2, 2],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0], [3, 3, 3, 3, 3]]
		example_matrix = np.array([np.array(xi) for xi in example_matrix])
		example_data_ids = [[1], [3], [1,1], [1,2], [2,3], [1,2,3]]
		example_weights = [None, None, [2,3], [3,2], [1,1], [2,3,4]]
		example_full_data_ids = np.array([1, 2, 3, 4])

		solution_matrix = \
		[[1.,         1.,         1.,         1.,         1.        ],\
		[0.,         0.,         0.,         0.,         0.        ],\
		[0. ,        0. ,        0.,         0.,         0.        ],\
		\
		[0.  ,       0.  ,       0.,         0.,         0.        ],\
		[0.   ,      0.   ,      0.,         0.,         0.        ],\
		[3.    ,     3.    ,     3.,         3.,         3.        ],\
		\
		[2.5    ,    2.5    ,    2.5,        2.5,        2.5       ],\
		[0.      ,   0.      ,   0. ,        0.,         0.        ],\
		[0.       ,  0.       ,  0. ,        0.,         0.        ],\
		\
		[1.5       , 1.5       , 1.5,        1.5,        1.5       ],\
		[2.,         2.,         2. ,        2.,         2.        ],\
		[0. ,        0. ,        0. ,        0.,         0.        ],\
		\
		[0.  ,       0.  ,       0. ,        0.,         0.        ],\
		[1.   ,      1.   ,      1. ,        1.,         1.        ],\
		[1.5   ,     1.5   ,     1.5,        1.5,        1.5       ],\
		\
		[0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667],\
		[2.        , 2.         ,2.,         2.,         2.        ],\
		[4.         ,4.         ,4.,         4.,         4.        ]]


		solution_matrix = np.array([np.array(xi) for xi in solution_matrix])

		appended_output, appended_ids = append_tracking_point_rows(example_matrix, example_full_data_ids, example_data_ids, weights=example_weights)

		self.assertTrue(isEqual(appended_output[appended_ids,:], solution_matrix, 1e-5)), "Append rows failed"

	def test_lstsqrs(self):
		V = [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]
		V = np.array([np.array(xi) for xi in V])
		target_data = np.random.rand(5,1)
		target_node_indices = range(5)
		A_r = reducedOrderApproximation(V, target_node_indices, nodes=target_data)

		self.assertTrue(isEqual(A_r, target_data, 1e-5)), "least squares approximation failed"

	def test_reduce_order(self):

		V = np.load("./TestData/example_basis.npy")
		target_data = np.load("./TestData/target_data.npy")
		target_data = target_data[:,-1]

		target_node_indices = np.load("./TestData/target_ids.npy")
		full_data = np.load("./TestData/full_data.npy")
		full_data = full_data[:,-1]

		full_node_num = full_data.shape[0]
		A_r = reducedOrderApproximation(V, target_node_indices, nodes=target_data)
		A_r = A_r[:full_node_num]

		self.assertTrue(isEqual(A_r, full_data, full_node_num*0.01)), "Reduced order approximation failed"


if __name__ == '__main__':
	unittest.main()