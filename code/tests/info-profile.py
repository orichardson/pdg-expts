
import sys; sys.path.append('../')

from dist import RawJointDist, CPT
from rv import Variable, binvar, Unit
import numpy as np
from pdg import PDG

M = PDG()

K = binvar("K")
X = binvar("X")
Y = binvar("Y")
M += X,Y,Z

globals().update(M.vars)

M += CPT.from_matrix(K, X, np.array([[.9,.1],[.1,.9]]) )
