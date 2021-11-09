%load_ext autoreload
%autoreload 2
from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import RawJointDist as RJD, CPT

from pdg.lib import A,B,C
import numpy as np

M = PDG()
M += A,B,C



############## B <---- A -----> C ##################
M += 'p', CPT.make_random(A,B)
M += 'q', CPT.make_random(A,C)

φ = M.factor_product()

mu_opt_torch = M._torch_opt_inc(0.01, constraint_penalty=0.01, ret_losses=False).npify()
# mu_opt_torch= mu_opt_torch[0].npify()

mu_opt_sqslp = M.optimize_score(0.01)
mu_mcmc = M.iter_GS_beta()

# SQSLP gives the right distribution, up to 1E-15.
mu_opt_sqslp // φ
mu_mcmc // φ

regions = [['p'],['q']]




############## A -----> B -----> C ##################
M2 = PDG()
M2 += 'p', CPT.make_random(A,B)
M2 += 'q', CPT.make_random(B,C)

φ2 = M2.factor_product()

mu2_opt_torch = M2._torch_opt_inc(0.01, constraint_penalty=0.01, ret_losses=False).npify()
mu2_opt_sqslp = M2.optimize_score(0.01)
mu2_mcmc = M2.iter_GS_beta(max_iters=8000,tol=1E-99, recalibrate=True)
mu2_mcmc_ordered = M2.iter_GS_ordered(recalibrate=True)
mu2_mcmc_ordered_backwards = M2.iter_GS_ordered(['q','p'],recalibrate=True)


# SQSLP gives the right distribution, up to 1E-15.
mu2_opt_sqslp // φ2
mu2_mcmc // φ2
mu2_mcmc_ordered // φ2
mu2_mcmc_ordered_backwards // φ2


M2.Inc(mu2_mcmc)
M2.Inc(mu2_mcmc_ordered)
M2.IDef(mu2_mcmc)
M2.IDef(mu2_mcmc_ordered)
M2.IDef(mu2_mcmc_ordered_backwards)
M2.IDef(φ2)


mu2_mcmc.I(A,C|B)
mu2_mcmc_ordered.I(A,C|B)


τ = M2.mk_edge_transformer('p')
M2.Inc(φ2)
τ(φ2) // φ2
M2.Inc(τ(φ2))

τ(mu2_mcmc) // mu2_mcmc

mu2_mcmc.info_diagram(A,B,C)
mu2_mcmc_ordered.info_diagram(A,B,C)

mu2_mcmc[C|B] - M2[C|B]
mu2_mcmc_ordered_backwards[C|B] - M2[C|B]
mu2_mcmc_ordered[C|B] - M2[C|B]

#########
# Notes: 
# |*> mu_mcmc[C|B] doesn't reflect q[C|B], even though there are no cycles,
# | and there's only one distribution on C. Why is this? Because half of
# | the samples have just resampled B from A,
# | 

# from collections import defaultdict
# counts = defaultdict(lambda: 0)
counts = np.ones(M2.dshape)
# μemperical = RJD(, M2.varlist)
μemperical = M2.genΔ(RJD.unif)

incs = []

for s in M2.MCMC(3000):
    # print(s)
    # counts[''.join(s.values())] += 1
    # print(tuple(v.ordered.index(s[v.name]) for v in M2.atomic_vars))
    counts[tuple(v.ordered.index(s[v.name]) for v in M2.atomic_vars)] += 1
    # counts /= counts.sum()
    μemperical.data = counts / counts.sum()
    
    incs.append(M2.Inc(μemperical).real)
    
# counts /= counts.sum()
incs = np.array(incs)
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

from matplotlib import pyplot as plt
plt.plot(smooth(incs,2))

print(incs)


μemperical[C,B|A]
φ2[C,B|A]
regions = [['p'],['q']]


########### A ----> B <------- C ############
M3 = PDG()
M3 += 'p', CPT.make_random(A,B)
M3 += 'q', CPT.make_random(C,B)
