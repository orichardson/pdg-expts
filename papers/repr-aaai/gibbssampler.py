# [ ] 
#BEGIN_FOLD 
%load_ext autoreload
%autoreload 2

import numpy as np
import math
from operator import mul

import sys
sys.path.append('/home/oli/Research/Joe/agent-goals/code')

def primes():
    i = 1
    # for counter in range(100):
    while(True):
        i += 1
        for j in range(2,min(int(math.sqrt(i))+1,i)):
            if i % j == 0:
                break
        else:
            yield i
            
#END_FOLD

from pdg import *
from rv import *
from dist import *

M = PDG()
# [ ]
variables = ['A', 'B', 'C', 'D']
V = { N : [N.lower()+str(i) for i in range(p)] for (N,p) in zip(variables, primes()) }
# V = { N : [N.lower()+str(i) for i in range(2)] for N in variables }
adj = {'AB', 'BD', 'CD', 'AC'}
Ed = set( (''.join(sorted(set(x for X in adj for x in X if Y in X if x != Y))),Y) for Y in variables )
# consistent DN from MRF
# P = { Y: np.random.random( (len(V[Y]), np.prod([len(V[X]) for X in PaY]) )) for PaY,Y in Ed } 
#rows, columns = outdim, indim

M = PDG()
for v in variables:
    M += Variable(V[v], name=v)
    
for PaY,Y in Ed:
    # M += CPT.make_stoch(M(' '.join(PaY)), M(Y), P[Y].T )
    M += CPT.make_random(M(' '.join(PaY)), M(Y) )

# CPT.from_matrix(M('A D'), M('B'), P['B'].T)
    

locals().update(**M.vars)
# CPT.make_stoch(B*C, A, P['A'].T)


μ1 = M.factor_product()
μ1.I(B | A,D)
μ1.I(B | A,D,C)

μ1[B | A]
M[A | B*C]
μ1[A | B*C*D]
μ1[A | B,C]

μ1.prob_matrix(A | C*B*B).shape
μ1.prob_matrix(A | C*B).shape
μ1.prob_matrix(A | B*C).shape
μ1.prob_matrix(A | B*C*D).shape
μ1.prob_matrix(A | B).shape
# [*(A*B*C*B).split()]

# Z = np.arange(3*2*2*6).reshape(3,2,2,6)
# np.einsum(Z, [0,1,1,2],[2,1,0])

# (B*C*B).structure[0].components
# (B*C*B).structure[1].components
# (A*B*C*D).ordered
# list(itertools.product(*(tuple(v.ordered) for v in [B,B])))

M.score(μ1)

μ1 = M.factor_product()
μ1.data.shape
μ2 = M.factor_product("atomic")
μ2.data.shape

M.score(RawJointDist.unif(M.atomic_vars))
# M.score(RawJointDist.unif(M.varlist))

np.allclose( μ2[B | A,C], μ1[B | A,C])
np.allclose( μ2[B,D | A,C], μ1[B,D | A,C])
np.allclose( μ2[A,B,C,D], μ1[A,B,C,D])
np.allclose(μ2.prob_matrix(A,B,C,D, B*C), μ2.prob_matrix(A,B,C,D, B*C))
np.allclose(μ2.prob_matrix(A,B,C,D, B*C, A*D), μ2.prob_matrix(A,B,C,D, B*C, A*D))

from scipy.optimize import minimize, LinearConstraint, Bounds
from torch import optim, autograd, tensor

import networkx as nx
nx.draw_networkx(M.graph, pos={'A': (1,1), 'B': (-1,1), 'C': (-1, -1), 'D' : (1,-1), 'B×C': (-0.5,0), 'A×D':(0.5,0)})

mscore = M._build_fast_scorer(gamma=1,repr='atomic')
mscore(μ2.data)

M.genΔ(RawJointDist.unif,'atomic').data.shape

M.score(μ1, gamma=1)
mscore(μ2.data)[0]

M._dist_repr = "raw"
M.varlist
μ1.I(A | B, C)
μ1.I(A | B, C, D)

μ1.I(A,D | B,C)
μ1.I(B,C | A,D)

np.allclose(μ1.data.sum(axis=(0,5,6)), μ2.data)

μ3 = M.optimize_score(gamma=0)

μ3[B | A,D]
M[B | A*D]

# optimize.minimize(mscore, μ1.data.copy())

# init = μ2.data.reshape(-1)
# req0 = (init == 0) + 0
# init.shape
# 
# opt1 = minimize(mscore, 
#     init +  np.random.rand(*init.shape)*1E-2 * (1-req0),
#     jac=True,
    # constraints = [LinearConstraint(np.ones(init.shape), 1,1), LinearConstraint(req0, 0,0)],
    # bounds = Bounds(0,1),
        # callback=(lambda xk,w: print('..', round(f(xk),2), end=' \n')),
    # method='trust-constr',
    # options={'disp':True}
    # )
#### NOW for the furn part.

p = RawJointDist.random(M.atomic_vars)
M2 = M.make_edge_mask(p)

# list(M2.cpds)
M2.score(p,gamma=0.01)
p2 = M2.optimize_score(gamma=0.01)

M2.score(p2, gamma=0.01)

p2[B|A,D]-p[B|A,D]
# These should be very close
p2.H(B | A,D)
p2.H(B | A,D,C)
# (here's the baseline; the first should ideally be the same and the second further than the above)
p.H(B | A,D)
p.H(B | A,D,C)

# Baseline:
p.H(C | A,D)
p.H(C | A,D,B)

p2.H(C | A,D)
p2.H(C | A,D,B)

Pr = p.prob_matrix
np.allclose(Pr(C|A,B) * Pr(A,B), Pr(A,B,C))
np.allclose(Pr(C|A,B) * Pr(A,B) * Pr(D | A,B,C), p.data)

# testing convergence 
Q = RawJointDist(Pr(C|A,B) * Pr(A,B,D), p.varlist)
np.allclose( Q.data, Q.prob_matrix(C |A,B) * Q.prob_matrix(A,B,D))

mscore = M2._build_fast_scorer(gamma=0,repr='atomic')
mscore(M.factor_product(repr='atomic').data)[1].max()

np.allclose(M2[B | A*D], p[B | A*D])
np.allclose(M2[C | A*D], p[C | A*D])
np.allclose(M2[A | B*C], p[A | B*C])
np.allclose(M2[D | B*C], p[D | B*C])
np.allclose(M2[A | A*D], p[A | A*D])
np.allclose(M2[D | A*D], p[D | A*D])
np.allclose(M2[B | B*C], p[B | B*C])
np.allclose(M2[C | B*C], p[C | B*C])

# M2._apply_structure()

inc = M2.Inc(p, True); inc

M2.Inc(M2.optimize_score(gamma=0, tol=1E-7))
M._opt_rslt.x
M2.Inc(M2.optimize_score(gamma=1))


M.Inc(μ2)
M.Inc(M.optimize_score(gamma=0))
