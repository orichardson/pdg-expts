#In [ ] 
%load_ext autoreload
%autoreload 2
from operator import mul

import sys
sys.path.append('/home/oli/Research/Joe/agent-goals/code')

from dist import RawJointDist as RJD
from lib import A,B,C,D
from lib.square import with_indeps, consist_with_P, P

consist_with_P.Inc(P)
μ = with_indeps.optimize_score(gamma=0.999999999, tol=1E-30)
P.I(A,B|C,D)
μ.I(A,B|C,D)

μ.I(B | A,D)
μ.I(B | A,D,C)

consist_with_P.score(P)
with_indeps.score(μ, gamma=0)
νo, iter_o = consist_with_P.iter_GS_ordered(tol=1E-30, store_iters=True)
νb = consist_with_P.iter_GS_beta(tol=1E-30)


consist_with_P.Inc(P)
consist_with_P.IDef(P)
consist_with_P.score(P, gamma=1)


round(0.123456789, 4)

import seaborn as sns
greens = sns.light_palette("green", as_cmap=True)
# νb[B|A].style.background_gradient(cmap=greens, axis=None)

(νo // νb)
## Wait are these not the same for consistent distributions ...?

[round(RJD(io.data, νb.varlist) // νb,5) for io in iter_o]
# it converges real fast, so that's not the problem...


(ν0 // μ)

consist_with_P.score(ν0,gamma=0)
consist_with_P.score(νb,gamma=0)



μ1 = M.factor_product()

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


M2.Inc(p2)

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
# The reason these dont work = 
# there are two copies of each edge labeled (π1) and (0), and so the
# cpd lookup is ambiguous and fails. To fix: stop duplicating these
# deterministic, auto-filled edges on mask creation.
np.allclose(M2[A | A*D], p[A | A*D])
np.allclose(M2[D | A*D], p[D | A*D])
np.allclose(M2[B | B*C], p[B | B*C])
np.allclose(M2[C | B*C], p[C | B*C])


inc = M2.Inc(p2, True); inc

p3 = M2.optimize_score(gamma=0, tol=1E-15)

mscore2 = M2._build_fast_scorer(gamma=0,repr='atomic')
M2.score(p3,gamma=0)
M2.Inc(p3)
mscore2(p3.data)[0]




# Try Gibbs Sampling
cpds = [*M2.edges('P')]
cpds

p3 = M2.optimize_score(gamma=0, tol=1E-20)
p4 = M2.iter_GS_ordered()
p5 = M2.iter_GS_beta()

np.real(M2.Inc(p5,True)[4:])
np.real(M2.Inc(p4,True)[4:])
np.real(M2.Inc(p3,True)[4:])
np.real(M2.Inc(p2,True)[4:])
np.real(M2.Inc(p,True)[4:])


M2.Inc(p4)
M2.Inc(p2)
M2.Inc(p3)
M2.Inc(p)

# The original distribution
p.I(A,D|B,C)
p.I(B,C|A,D)

# The result of Ordered Gibbs Sampling
p4.I(A,D|B,C)
p4.I(B,C|A,D)

# The result of Beta Gibbs Sampling
p5.I(A,D|B,C)
p5.I(B,C|A,D)

# Our optimal score
p3.I(A,D|B,C)
p3.I(B,C|A,D)

M3 = PDG()

# np.allclose(cpds[0], p2.broadcast(cpds[0]))
(p2.prob_matrix(B|B) * p2.prob_matrix(A,B,C,D,B*D)).sum()
np.allclose((p2.prob_matrix(B|B) * p2.prob_matrix(A,B,C,D) ), p2.prob_matrix(A,B,C,D))
np.allclose((p2.prob_matrix(B|A,C,D) * p2.prob_matrix(A,C,D) ), p2.prob_matrix(A,B,C,D))
np.allclose((p2.prob_matrix(B|B*C) * p2.prob_matrix(A,B,C,D) ), p2.prob_matrix(A,B,C,D))


# In [ ]
    M3 = PDG()
    M3 += A,B,C
    q = RawJointDist.random([A,B,C])
    M3 += q[B | A,C]
    M3 += q[A | B]
    M3 += q[C | B]


    [*M3.edges('Xn Yn')]

    q2 = M3.optimize_score(gamma=0, tol=1E-20)
    q3 = M3.optimize_score(gamma=0.0001, tol=1E-20)
    q4 = M3.iter_GS_ordered()
    q5 = M3.iter_GS_beta()
    qFP = M3.factor_product(repr='atomic')

    np.real(M3.Inc(qFP,True)[2:])
    np.real(M3.Inc(q5,True)[2:])
    np.real(M3.Inc(q4,True)[2:])
    np.real(M3.Inc(q3,True)[2:])
    np.real(M3.Inc(q2,True)[2:])
    np.real(M3.Inc(q,True)[:3])


    [ np.sum((qi.data - q.data)**2) for qi in [q2,q3,q4,q5,q6,q]]

    # In [ ]
    # The original distribution
    q.I(A | B,C)
    q.I(B,C | A)

    # The result of Ordered Gibbs Sampling
    q4.I(A | B,C)
    q4.I(B,C | A)

    # The result of Beta Gibbs Sampling
    q5.I(A | B,C)
    q5.I(B,C | A)

    # Factored distribution
    qFP.I(A | B,C)
    qFP.I(B,C | A)


### EXPERIMENT: AN ACTUALLY CONSISTENT DN
    M4 = PDG()
    M4 += CPT.make_random(A,B)
    M4 += CPT.make_random(A,C)
    
### EXPERIMENT: VISUALIZE BEHAVIOR
# Collect iterates of both opt procedure, and gibbs sampling
# Also bundle with original vector p.

    # M4.iter_GS_beta(store_iters=True)
    from collections import defaultdict

    D = M4.all_dists()
    
    idx = defaultdict(list)
    for i,(tag,mat) in enumerate(D.items()):
        if 'opt' in tag: 
            idx['opt'].append(i)
        
        if 'π' in tag:
            idx['π'].append(i)
        
    bigmatrix = np.array(list(D.values()))
    bigmatrix.shape
    
    from sklearn.decomposition import PCA
    
    tovis = PCA(n_components=2)
    tovis.fit(bigmatrix)
    
    
    import matplotlib
    matplotlib.use('Qt5Agg')
    # This should be done before `import matplotlib.pyplot`
    # 'Qt4Agg' for PyQt4 or PySide, 'Qt5Agg' for PyQt5
    import matplotlib.pyplot as plt
    X,Y = tovis.transform(bigmatrix).T

    fig, ax = plt.subplots()
    ax.scatter(X,Y)
    
    for i,tag in enumerate(D.keys()):
        ax.annotate(tag, (X[i], Y[i]) )


# # FIRST WITH PCA
# def pcaplot():
#     from sklearn.decomposition import PCA
#     tovis = PCA(n_components=2)
#     tovis.fit(bigmatrix)


#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots()
#     # # ax.plot(*tovis.transform(iter0pt).T, '.k-')
#     # # ax.plot(*tovis.transform(iter1pt).T, '.y-')
#     plt.scatter(*tovis.transform(iter0pt).T, cmap='Purples', c=idx0pt)
#     plt.scatter(*tovis.transform(iter1pt).T, cmap='Oranges', c=idx1pt)

#     # ax.plot(*tovis.transform(iter1pt).T, '.k-')

#     ax.plot(*tovis.transform(Ppt).T, 'ob')
#     ax.plot(*tovis.transform(φpt).T, 'or')
#     plt.show()
# %matplotlib notebook
