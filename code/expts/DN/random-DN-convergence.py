from pdg.pdg import PDG
from pdg.rv import Variable as RV
from pdg.dist import RawJointDist as RJD, CPT

import numpy as np


A = RV.alph("A", 2)
B = RV.alph("B", 2)
C = RV.alph("C", 2)

M = PDG()
M += A, B, C

p = CPT.from_matrix(A, B, np.array([[0.7, 0.3], [0.8, 0.2]]))
q = CPT.from_matrix(B, C, np.array([[0.7, 0.3], [0.8, 0.2]]))
r = CPT.from_matrix(C, A, np.array([[0.7, 0.3], [0.8, 0.2]]))

p@q


M_pqr = PDG()
M_pqr += p,q,r

# μ = M_pqr._torch_opt_inc()
μ = M_pqr.optimize_score(0)
M_pqr.IDef(μ)


μ.info_diagram(A,B,C)





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
