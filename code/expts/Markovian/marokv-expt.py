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
A,B,C = Var.alph("A", 3), Var.alph("B", 2),  Var.alph("C", 3)
M += A,B,C

# Add p(B|A), q(C|B), r(A|C).
M += 'p', CPT.make_random(A,B)
M += 'q', CPT.make_random(B,C)
M += 'r', CPT.make_random(C,A)

# M.update_all_weights(0.1, 1)


# μ1 = M.optimize_score(1E-12)
μ1 = M.optimize_score(0)
ϕ = M.factor_product()
μ2 = M._torch_opt_inc(0, extraTemp=0, iters=3000, constraint_penalty=0, \
    representation="gibbs", lr=3E-2).npify()


M.Inc(μ1)
M.Inc(μ2)
M.IDef(μ1)
M.IDef(μ2)
M.Inc(ϕ)
M.IDef(ϕ)


μiter = M.iter_GS_beta()
M.Inc(μiter)
M.IDef(μiter)



# Now, try to match cpds via factors.

M._torch_opt_inc(extraTemp=0, constraint_penalty=0, gamma=0)
μ_cvr = M.optimize_via_FG_cover(iters=5000).npify()
M.Inc(μ_cvr), M.IDef(μ_cvr)


μ_cvr // μ1
μ2 // μ1
μ2 // μ_cvr
μ_cvr // μ2

μ2.data.min()
μ_cvr.data.min()

# Question: why is μ1//μ2 less than zero, given that both sum to 1?
μ1 // μ2
## Possible answer: has zeros and 1E-26 values. 
((μ1.data - μ2.data)**2).sum()
((μ1.data - μ_cvr.data)**2).sum()
((μ2.data - μ_cvr.data)**2).sum()
