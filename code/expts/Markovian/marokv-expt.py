#### QUESTION ####
"""
Are there cpds for 
"""



###
%load_ext autoreload
%autoreload 2
import sys, os
# import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

os.getcwd()

sys.path.append("..")

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import RawJointDist as RJD, CPT

import numpy as np

M = PDG()
A,B = Var.alph("A", 3), Var.alph("B", 2)
M += CPT.make_random(A,B)
M += CPT.make_random(A,B)

M.update_all_weights(0.1, 1)

M.edgedata['A','B', 'p1']

list(M.edges("l"))
M['p2']
