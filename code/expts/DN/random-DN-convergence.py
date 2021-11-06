%load_ext autoreload
%autoreload 2

from pdg.pdg import PDG
from pdg.rv import Variable as RV
from pdg.dist import RawJointDist as RJD, CPT

import numpy as np


A = RV.alph("A", 2)
B = RV.alph("B", 2)
C = RV.alph("C", 2)

M = PDG()
M += A, B, C

# p = CPT.from_matrix(A, B, np.array([[0.7, 0.3], [0.8, 0.2]]))
# q = CPT.from_matrix(B, C, np.array([[0.7, 0.3], [0.8, 0.2]]))
# r = CPT.from_matrix(C, A, np.array([[0.7, 0.3], [0.8, 0.2]]))
p = CPT.from_matrix(A, B, np.array([[0.7, 0.3], [0.4, 0.6]]))
q = CPT.from_matrix(B, C, np.array([[0.7, 0.3], [0.4, 0.6]]))
r = CPT.from_matrix(C, A, np.array([[0.7, 0.3], [0.4, 0.6]]))

p@q


M_pqr = PDG()
M_pqr += p,q,r

M_pqr.update_all_weights(a=0.1)

μ = M_pqr._torch_opt_inc(gamma=1E-2,optimizer='SGD', ret_losses=False, iters=2000)
μ.npify()[C|B]
μ2 = M_pqr.optimize_score(1E-8)
# μ = 
M_pqr.IDef(μ)


μ.npify().info_diagram(A,B,C)
μ[A,B,C]
μ2[B|A]

### Trying to find 3 factors that generate μ.data:
import torch
ϕ1 = torch.zeros((2,2,1), requires_grad=True)
ϕ2 = torch.zeros((2,1,2), requires_grad=True)
ϕ3 = torch.zeros((1,2,2), requires_grad=True)

μt = μ.torchify()

ozr = torch.optim.Adam([ϕ1, ϕ2, ϕ3], lr=1E-3)
losses = []
for it in range(2000):
    ozr.zero_grad()
    density = torch.exp(ϕ1 + ϕ2 + ϕ3) 
    newdist = density / density.sum()
    loss = (μt.data * (torch.log(μt.data) - torch.log(newdist))).sum()
    loss.backward()
    losses.append(loss.detach())
    ozr.step()

plt.plot(losses)    
print(loss)

μt.data
newdist.data
####################################################


fp = M_pqr.iter_GS_beta()
fp[B|A]
fp[C|B]
fp[A|C]

M_pqr.IDef(fp)
fp.info_diagram(A,B,C)



# Create factor graph (A,B) * (B,C) * (C,D)
def randfactor(*shape):
    return np.random.exponential(size=shape)
D = RJD(randfactor(1,2,2)* randfactor(2,1,2)*randfactor(2,2,1), [A,B,C])
D.info_diagram(A,B,C)

γs = [0, 1E-12, 1E-8, 1E-4, 1E-2, 1E-1, 0.5, 1, 1.5, 10, 100, 1E3, 1E5]
μs = [ M_pqr.optimize_score(γ) for γ in γs ]
M_pqr.varlist
μs_torch = [ M_pqr._torch_opt_inc(γ, ret_losses=False) for γ in γs ]

Inc, IDef = M_pqr.Inc, M_pqr.IDef
for i,γ in enumerate(γs):
    print('γ: {:.2e} |  INC:  t {.real:.3e};  s {.real:.3e}; |  IDEF: t {:.3e};   s {:.3e} | {:.3f} {:.3f}'
        .format(γ, Inc(μs[i]), Inc(μs_torch[i].npify()), IDef(μs[i]), IDef(μs_torch[i].npify()), μs[i].I  ))


M_pqr.IDef(fp)

from matplotlib import pyplot as plt

X,Y = zip(*[(M_pqr.Inc(m).real, M_pqr.IDef(m)) for m in μs])
plt.plot( X, Y)
plt.scatter(X, Y, s=50, c=np.arange(len(X)), cmap="Blues")










################## IDEA: CHAIN LENGTH? ################
# N = 4

vals = []
for N in range(1, 10):
    M = PDG()
    Xs = [RV.alph("X%d"%(i),2) for i in range(N)]

    for i in range(N):
        M += Xs[i]
        M += { 'α' : 1,  'beta': 1 ,
            'cpd' : CPT.from_matrix(Xs[i], Xs[(i+1)%N], np.array([[0.7, 0.3], [0.8, 0.2]])) }
    
    d = M.iter_GS_beta(max_iters=600*N).prob_matrix(Xs[1%N], given=[Xs[0]])
    # print(d)
    vals.append( d.flatten()[0])


#%matplotlib tk
# RESULTS OF PLOTTING: damped oscillation around 0.713. 
plt.plot(vals)
%matplotlib inline

# For 1
M.iter_GS_beta()[Xs[0]|Xs[0]]
