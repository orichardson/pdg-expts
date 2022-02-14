import numpy as np
from dist import D_KL, zz1_div,z_mult

np.ma.where([True, False], 1, [3,7])

zz1_div(np.array([0,1]), np.array([0,2]))
D_KL(np.array([0,1]), np.array([0,1]))
D_KL([.5,.5], [.5,.5])
z_mult(np.array([0,1,2]), zz1_div(np.array([4,1,2]), np.array([0,1,3])))
