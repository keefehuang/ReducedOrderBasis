from matplotlib.pyplot import *
import numpy as np 
import scipy as sp 


# results = np.array([[2, 0.03371361420473884],
# [3, 0.020433970492624288],
# [4, 0.013127366324325809],
# [6, 0.005826724236309396],
# [9, 0.0027221953556894967],
# [13, 0.0017430856867353091],
# [19, 0.0007497787353923833],
# [28, 0.0002917117107054171],
# [42, 0.0001354664301531366],
# [63, 5.7971162502249624e-05],
# [94, 1.6937953977177836e-05],
# [141, 3.228674426757337e-06],
# [211, 3.205522589341942e-07]])


# fig, ax = subplots()
# ax.semilogy(results[:,0], results[:,1])
# title('RMS Error vs. # ROBs')
# ylabel('RMS Error')
# xlabel('# Reduced Order Basis')
# show()

fig, ax = subplots()

results = np.load('errors_trial3.npy')

ax.semilogy(results[:,0], results[:,1])
title('RMS Error vs. # ROBs for Reduced Nodes')
ylabel('RMS Error')
xlabel('# Reduced Order Basis')
show()