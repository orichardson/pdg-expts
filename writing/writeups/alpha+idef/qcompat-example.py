import numpy as np

%cd ../code
from pdg.dist import RawJointDist as RJD
from pdg.rv import Variable as Var, binvar

nuU = np.array([
    1, # 0 00 00       # 000
    1, # 0 00 01       # 000
    0, # 0 00 10       # 001
    0, # 0 00 11       # 001
    1, # 0 01 00       # 000
    1, # 0 01 00       # 000
    0, # 0 01 10       # 001
    0, # 0 01 11       # 001
    1, # 0 10 00       # 010
    0, # 0 10 01       # 011
    1, # 0 10 10       # 010 
    0, # 0 10 11       # 011
    1, # 0 11 00       # 010
    0, # 0 11 01       # 011
    1, # 0 11 10       # 010
    0, # 0 11 11       # 011
    #
    0, # 1 00 00       # 100
    0, # 1 00 01       # 100
    1, # 1 00 10       # 101
    1, # 1 00 11       # 101
    0, # 1 01 00       # 110
    1, # 1 01 01       # 111
    0, # 1 01 10       # 110
    1, # 1 01 11       # 111
    0, # 1 10 00       # 100
    0, # 1 10 01       # 100
    1, # 1 10 10       # 101
    1, # 1 10 11       # 101
    0, # 1 11 00       # 110
    1, # 1 11 01       # 111
    0, # 1 11 10       # 110
    1, # 1 11 11       # 111
])

FA = binvar("FA")
FB0 = binvar("FB0_")
FB1 = binvar("FB1_")
FC0 = binvar("FC0_")
FC1 = binvar("FC1_")

A = binvar("A"); B = binvar("B"); C = binvar("C")
 
# U = FA & FB0 & FB1 & FC0 & FC1
U = Var.product([FA, FB0, FB1, FC0, FC1])

nuU = nuU.reshape((2,)*5) / nuU.sum()

νU = RJD(nuU, [*U.atoms])

0b010
