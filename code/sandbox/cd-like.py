x = 3

from pdg import PDG, RV as Var, RJD, CPT

M = PDG()

A = Var.alph("A", 3)
B = Var.alph("B", 3)

M += CPT.make_random(A, B)

M.draw()

### SCRATCH
# μ = M.optimize_via_FG_cover()
# [*M.edges("P")]
# μ[B|A]
# M['p1']
import numpy as np
f1 = np.random.rand(2,3,5,7)
f2 = np.random.rand(1,5,7)
(f1*f2).shape

from pdg.fg import FactorGraph
F = FactorGraph([f1,f2])

F.vars
F.dist


from pdg.lib import A,B,C,D
M = PDG()
M += A, B, C, D
M += CPT.make_random(A, B)
M += CPT.make_random(B, C & D)
M += CPT.make_random(D&A, C)

assert np.allclose(M.to_FG().dist.data, M.factor_product().data)

### Optimiziation
