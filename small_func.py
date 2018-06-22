import numpy as np


def rearange_xyz(A):
    """
    Rearange array or a vector from [-x-, -y-, -z-]  into
    [x1,y1,z1, ... , xn, yn, zn] format.

    Notes
    -----
    /

    Input
    -----
    A	:	array-like, shape(m,k)
    	Array with points stored in columns in [-x-, -y-, -z-] format.

    Output
    ------
    A_new	:	array-like, shape(m,k)
    	Array with points stored in columns in [x1,y1,z1, ... , xn, yn, zn] format.
    """

    A_dim   = np.shape(A)
    num_pts = int( A_dim[0]/3 )
    # vector
    if len(A_dim) < 2:
        A_new   = np.empty( [A_dim[0]] )
        for i in range(num_pts):
            A_new[3*i]   = A[i]
            A_new[3*i+1] = A[i +   num_pts]
            A_new[3*i+2] = A[i + 2*num_pts]
    # array
    else:
        A_new   = np.empty([A_dim[0], A_dim[1]])
        for i in range(num_pts):
            A_new[3*i,:]   = A[i,:]
            A_new[3*i+1,:] = A[i +   num_pts, :]
            A_new[3*i+2,:] = A[i + 2*num_pts, :]
    return A_new

def rearange_xxx(A):
    """
    Rearange array or a vector from [-x-, -y-, -z-]  into
    [x1,y1,z1, ... , xn, yn, zn] format.

    Notes
    -----
    /

    Input
    -----
    A	:	array-like, shape(m,k)
    	Array with points stored in columns in [-x-, -y-, -z-] format.

    Output
    ------
    A_new	:	array-like, shape(m,k)
    	Array with points stored in columns in [x1,y1,z1, ... , xn, yn, zn] format.
    """

    A_dim   = np.shape(A)
    num_pts = int( A_dim[0]/3 )
    # vector
    if len(A_dim) < 2:
        A_new   = np.empty( [A_dim[0]] )
        for i in range(num_pts):
            A_new[i]             = A[3*i]
            A_new[i +   num_pts] = A[3*i+1]
            A_new[i + 2*num_pts] = A[3*i+2]
    # array
    else:
        A_new   = np.empty([A_dim[0], A_dim[1]])
        for i in range(num_pts):
            A_new[i,:]              = A[3*i,:]
            A_new[i +   num_pts, :] = A[3*i+1,:]
            A_new[i + 2*num_pts, :] = A[3*i+2,:]
    return A_new
