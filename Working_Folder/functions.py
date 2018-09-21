import pickle
import numpy as np
from preprocessing import *
import scipy

def simple_average(tracking_points, weights):
	summed_points = None
	for point in range(tracking_points.shape[0]//3):
		if summed_points is None:
			summed_points = tracking_points[[point*3, point*3+1, point*3+2], :]
		else:
			summed_points += tracking_points[[point*3, point*3+1, point*3+2], :]
	return summed_points/(tracking_points.shape[0]/3)

def weighted_average(tracking_points, weights):
	summed_points = None
	for i, point in enumerate(range(tracking_points.shape[0]//3)):
		if summed_points is None:
			summed_points = weights[i]*tracking_points[[point*3, point*3+1, point*3+2], :]
		else:
			summed_points += weights[i]*tracking_points[[point*3, point*3+1, point*3+2], :]
	return np.array(summed_points/(tracking_points.shape[0]/3))

weights_0 = None
weights_1 = [31.6, 42.6, 15.3]
# tracking_points = [[6180, 6135, 14512], [20772, 14329, 13567]]

functions = {0 : [weighted_average, weights_0], 1 : [weighted_average, weights_1]}

functions = {	0 : [simple_average, weights_0],
1 : [simple_average, weights_0],
2 : [simple_average, weights_0],
3 : [simple_average, weights_0],
4 : [simple_average, weights_0],
5 : [simple_average, weights_0],
6 : [simple_average, weights_0],
7 : [simple_average, weights_0],
8 : [simple_average, weights_0],
9 : [simple_average, weights_0],
10 : [simple_average, weights_0],
11 : [simple_average, weights_0],
12 : [simple_average, weights_0],
13 : [simple_average, weights_0],
14 : [simple_average, weights_0],
15 : [simple_average, weights_0],
16 : [simple_average, weights_0],
17 : [simple_average, weights_0],
18 : [simple_average, weights_0],
19 : [simple_average, weights_0],
20 : [simple_average, weights_0],
21 : [simple_average, weights_0],
22 : [simple_average, weights_0],
23 : [simple_average, weights_0],
24 : [simple_average, weights_0],
25 : [simple_average, weights_0],
26 : [simple_average, weights_0],
27 : [simple_average, weights_0],
28 : [simple_average, weights_0],
29 : [simple_average, weights_0],
30 : [simple_average, weights_0],
31 : [simple_average, weights_0],
32 : [simple_average, weights_0],
33 : [simple_average, weights_0],
34 : [simple_average, weights_0],
35 : [simple_average, weights_0],
36 : [simple_average, weights_0],
37 : [simple_average, weights_0],
38 : [simple_average, weights_0],
39 : [simple_average, weights_0],
40 : [simple_average, weights_0],
41 : [simple_average, weights_0],
42 : [simple_average, weights_0],
43 : [simple_average, weights_0],
44 : [simple_average, weights_0],
45 : [simple_average, weights_0],
46 : [simple_average, weights_0],
47 : [simple_average, weights_0],
48 : [simple_average, weights_0],
49 : [simple_average, weights_0],
50 : [simple_average, weights_0],
51 : [simple_average, weights_0],
52 : [simple_average, weights_0],
53 : [simple_average, weights_0],
54 : [simple_average, weights_0],
55 : [simple_average, weights_0],
56 : [simple_average, weights_0],
57 : [simple_average, weights_0],
58 : [simple_average, weights_0],
59 : [simple_average, weights_0],
60 : [simple_average, weights_0],
61 : [simple_average, weights_0],
62 : [simple_average, weights_0],
63 : [simple_average, weights_0],
64 : [simple_average, weights_0]
			}

def run_function(func, tracking_points, weights):
	return func(tracking_points, weights)
	