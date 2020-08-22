%load_ext autoreload
%autoreload 2
# should be started in root directory
import sys; sys.path.append('../')



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dist import RawJointDist, CPT
from rv import Variable, binvar, Unit
from pdg import *


# In[ ] 

M = PDG()
PS = binvar('PS')
S = binvar('S')
SH = binvar('SH')
C = binvar('C')


M += CPT.from_ddict(Unit, PS, {'⋆': 0.3})
M += CPT.from_ddict(PS, S, { 'ps': 0.4, '~ps' : 0.2})
M += CPT.from_ddict(PS, SH, { 'ps': 0.8, '~ps' : 0.3})
M += CPT.from_ddict(S * SH, C, 
    { ('s','sh') : 0.6, ('s','~sh') : 0.4,
      ('~s','sh'): 0.1, ('~s','~sh'): 0.01} )
      

# Right now it's just a the BN distribution we expect
T = binvar("T"); M += T
mu1 = M.factor_product()    # Variable added first
mu1.H(...)
M.score(mu1)
f = M._build_fast_scorer() # [1, -1, 0, 0]
init = RawJointDist.unif(M.varlist).data.reshape(-1)



# In[ ]
import torch
from torch import optim, autograd, tensor


optim.

from scipy.optimize import minimize, LinearConstraint, Bounds

req0 = (mu1.data.reshape(-1) == 0) + 0
mu1.data.reshape(-1).dot(req0)
opt1 = minimize(f, 
    mu1.data.reshape(-1) +  np.random.rand(*init.shape)*1E-2,
    constraints = [LinearConstraint(np.ones(init.shape), 1,1), LinearConstraint(req0, 0,0)],
    bounds = Bounds(0,1),
        # callback=(lambda xk,w: print('..', round(f(xk),2), end=' \n')),
    method='trust-constr',
    options={'disp':True}) ;
# Number of iterations: 531. Function evals: 6811. 


# cb = lambda xk, optr : print('..', round(or.fun,2), end='  '))
opt1.fun
opt1.fun
opt1.fun
np.sum((opt1.x-mu1.data.reshape(-1))**2)
opt1.x.sum()
mu2 = RawJointDist(opt1.x.reshape(*M.dshape) / opt1.x.sum(),M.varlist) 
mu2[PS]
mu1[PS]

opt1
(mu1.data == 0).sum()

f(init)
f(mu1.data.reshape(-1))
f(mu1.data.reshape(-1) + np.random.rand(*init.shape)*1E-2)
# opt1 = M.optimize_score()

# In[ ]

# CPT added
T2C =  CPT.from_ddict(T, C, { 't' : .3, '~t' : .05}); M += T2C
