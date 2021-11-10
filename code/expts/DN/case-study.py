
# %load_ext autoreload
# %autoreload 2

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import RawJointDist as RJD, CPT
from pdg.lib import A,B,C,D

E = Var.alph("E", 2)

import numpy as np

from operator import mul
from functools import reduce

def rollstr(strinput,k=1):
    k = k%len(strinput)
    return strinput[k:]+strinput[:k]

##### THe PDG ℳ is a pentagon. ####
M = PDG()
M += A,B,C,D,E
# M += "p", CPT.make_random(A,B)
# M += "q", CPT.make_random(A,B)

# adj = {'AB', 'BD', 'CD', 'DE'}
varletters = ''.join(X.name for X in M.varlist)
adj = set( c1+c2 for c1,c2 in zip(varletters, rollstr(varletters) ))

factors = [np.random.random(tuple((len(v) if v.name in ed else 1) for v in M.varlist)) for ed in adj]

P = RJD(reduce(mul, factors), M.varlist).normalize()

# P.I(A,D | B,C)
Ed = [ ( M(' '.join(sorted(x for X in adj for x in X if Y in X if x != Y))),   M(Y) ) for Y in varletters ]
for PaY,Y in Ed:
    M += P[Y | PaY]

M.Inc(P).real

#### Run the Gibbs Sampling Procedure ##########
μGS = M.iter_GS_ordered(max_iters=1000)

μGS // P,  P // μGS  # The two distributions should be identical.

product, norm_Z = M.factor_product(return_Z=True)

# ... also run it starting from the factor product, and get a trace.
μGS_fp, μGS_fp_iters = M.iter_GS_ordered(
    counterfactual_recalibration=True, max_iters=1000, init=product.clone(), store_iters=True)

# IDef should be D(P || product) - \log Z ...
assert( np.abs((P // product - np.log(norm_Z))/np.log(2) - M.IDef(P)) < 1E-10 )

###### What's the Dimension of the Consistent Distributions? ####
from numpy.linalg import svd
from matplotlib import pyplot as plt

dists = M.random_consistent_dists(200)
%matplotlib inline
plt.semilogy(sorted([1E-16+M.Inc(d).real for d in dists]),'-o')

consistmatrix = np.stack(tuple(d.data.reshape(-1) for d in dists))
consistmatrix.shape
U,Σ,VT = svd(consistmatrix)

from pdg.distviz import pca_view
def incidef(data):
    dist = M.genΔ()
    toret = np.zeros((len(data), 2))
    for i,d in enumerate(data):
        dist.data = d.reshape(M.dshape)
        toret[i,:] = [np.log(M.Inc(dist).real), M.IDef(dist)]
        # toret[i,:] = [M.Inc(dist).real, -dist.H(...)]
    return toret

randomdists = [RJD.random(M.varlist) for i in range(100)]

%matplotlib tk
## pca=
pca_view(random_consist=dists, 
    factor_prod = [product], 
    correct=[P], 
    GS_init_ϕ=[μGS_fp],
    truly_random=randomdists, 
    GS_init_ϕ_trace= μGS_fp_iters[:10]+μGS_fp_iters[10::30],
    arrows=False, transform=incidef)
plt.xlabel("Inconsistency")
plt.ylabel("Information Deficiency ($\\alpha=1$)")
## pca.transform(product.data.reshape(1,-1))

# It's much closer to the product than all the others.
μGS // product - min( d // product for d in dists) 







# random edges
# for PaY,Y in Ed:
#     with_rand_cpts += CPT.make_random(PaY, Y)
#     consist_with_P += P[Y | PaY]
# _prodpfactors = consist_with_P.factor_product()
# from numpy.random import choice
# 
# for PaY,Y in Ed:
#     # remove a random parent
#     parent_names = set(PaY.name.split('×'))
#     n2kill = choice([*parent_names])
#     SubPaY = _base( ' '.join(parent_names - { n2kill } ))
#     missing_parents += P[Y | SubPaY]
