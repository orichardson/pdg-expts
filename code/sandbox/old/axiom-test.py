%load_ext autoreload
%autoreload 2
import sys; sys.path.append('../')

from dist import RawJointDist, CPT
from rv import Variable, binvar, Unit
import numpy as np
from pdg import PDG

M = PDG()

A = binvar("A")
B = binvar("B")
C = binvar("C")
M += A, B, C


# First I do this with numpy
pA = np.array([0.5,0.5]);
pB_A = np.array([[0.5,0.5],[.5,.5]]);


# M += CPT.from_matrix(A, B, np.array([[.6,.4],[.4,.6]]) )
# M += CPT.from_matrix(B, C, np.array([[.6,.4],[.4,.6]]) )
# M += CPT.from_matrix(Unit, A, np.array([[0.55, 0.45]]))

# xxorkdict = { (x,k): ('~' if ((x[0]=='~')^(k[0]=='~')) else '')+"x⊕k" for x,k in X&K}
# M += CPT.det(X&K, binvar("X⊕K"), xxorkdict)
# yxorkdict = { (y,k): ('~' if ((y[0]=='~')^(k[0]=='~')) else '')+"y⊕k" for y,k in Y&K}
# M += CPT.det(Y&K, binvar("Y⊕K"), yxorkdict)
# allxordict = { (x,y,k): ('~' if ((y[0]=='~')^(k[0]=='~')^(x[0]=='~')) else '')+"x⊕y⊕k" for x,y,k in Variable.product(X,Y,K)}
# M += CPT.det(Variable.product(X,Y,K), binvar("X⊕Y⊕K"), allxordict)
# M.draw()
# 
# μ = M.factor_product()
# 
# XxorK = M.vars['X⊕K']
# YxorK = M.vars['Y⊕K']
# μ.H(XxorK)
# μ.H(X)
# μ.H(K)
# 
# μ.I(XxorK, YxorK, X)
# 
# μ.info_diagram(X,K,Y)
# μ.info_diagram(XxorK&K, K, YxorK&K)
# μ.info_diagram(XxorK, M.vars['X⊕Y⊕K'],YxorK)
# μ.info_diagram(X, M.vars['X⊕Y⊕K'],Y)
# 
# 
# A,B,C = binvar("A"), binvar("B"), binvar("C")
# # np.array([[[1,1],[1,1]], [[1,1],[1,1]]]).shape
# D = +RawJointDist( np.array([[[.7,.3],[.3,.7]], [[.3,.7],[.7,.3]]]), [A,B,C]);D.info_diagram(A,B,C)
# 
# # # For some reason, the interaction information is basically always negative...
# RawJointDist.random([A,B,C]).info_diagram(A,B,C)
# RawJointDist.unif([A,B,C]).info_diagram(A,B,C)
