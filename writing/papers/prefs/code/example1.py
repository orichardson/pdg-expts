# import os
# print(os.getcwd())
%load_ext autoreload
%autoreload 2

"""
There are three kinds of matrices to model:
    - preference matrices (or tensors! simplices! simplicial complexes!) 
    - transition matrices (links between)
    - distance matrices 
        ? is this a symmetrized preference matrix?
"""

from domains import *

A = Dom.empty([""])
A

np.random.normal()

A = Dom.randu("A", 4)
B = Dom.randu("B", 2)
