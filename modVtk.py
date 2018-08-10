import numpy
from vtk import *
from vtk.util import numpy_support as VN
from qd.cae.dyna import Binout
from Binout_reading import binout_reading
import small_func

filename = "/home/keefe/Documents/BMW/HiWi/Code/ReducedOrderBasis/Visualization/VTK_IN/Bumper/Bumper_0.vtk"
reader = vtkUnstructuredGridReader()
reader.SetFileName(filename)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()
points = reader.GetOutput().GetPoints().GetData()
data = reader.GetOutput()
# print("Number of Points: " ,  points.GetNumberOfPoints())

print(points)

points.PrintSelf()

# binout_coordinates 	= binout_reading("bumper.binout", False, 'coordinates')
# binout_ids 			= binout_reading("bumper.binout", False, 'ids')
# binout_coordinates 	= small_func.rearange_xyz(binout_coordinates)[:,0].reshape((-1,3))

# print(binout_ids.shape)

# for ids in binout_coordinates[:10]:
# 	print(ids)


# i = 0
# while(i < 10) :
#   print("Next Point : %d, %s " % (i + 1, points.GetPoint(i)))
#   i = i + 1


# print(points)


# print(data)

# w = VtkFile("/home/keefe/Documents/BMW/HiWi/Code/ReducedOrderBasis/Visualization/VTK_IN/Bumper/Bumper_1.vtk", VtkUnstructuredGrid)

# w.openGrid()
# w.openElement("Points")

# print()
