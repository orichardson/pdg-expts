import numpy as np
# p = np.array([0.6, 0.4, 0])
# r = np.array([0.3, 0.2, 0.5])
# q = np.array([0, 0.8, 0.2])
# s = np.array([0.2, 0.64, 0.16])
# # q = np.array([0, 0, 1.0])
# # s = np.array([0.125, 0.125, 0.75])
# # s = np.array([0.75, 0.125, 0.125])
# basis = np.array([[-1., -1], [1, -1.], [0, 0.5]])

# p = np.array([0.6, 0.4, 0, 0])
# r = np.array([0.3, 0.2, 0.2, 0.3])
# q = np.array([0,0, 0.8, 0.2])
# s = np.array([0.15, 0.05, 0.64, 0.16])

p = np.array([0.6, 0.3, .1, 0])
r = np.array([0.3, 0.15, 0.05, 0.5])
q = np.array([0, 0.1, 0.7, 0.2])
s = np.array([0.2, 0.08, 0.56, 0.16])
# r = s = (p + q) / 2

basis = np.array([[-1., -1], [1, -1], [-0.2, 0.5], [0.1, -0.6]])

A = (p > 0).astype(float); B = (q > 0).astype(float)


%matplotlib qt

def cond(mu, evt):
    pre = mu * evt
    return pre / pre.sum()

# A = np.array([1.,1.,0]); B = np.array([0, 1., 1.])


from matplotlib import pyplot as plt


c = np.linspace(0,1,500)



# for β,ζ in [(0,0), (1,1), (2,2), (3, 3), (3, 0), (3,1), (3,2), (0,3), (0,2), (0,1) ]:
# for k in [0,2,10,20,30,50,80,100, 200, 300, 500, 700]:
for k in [0,200]:
    β, ζ = k, k
    zs = cond(r, np.exp(β*A)).reshape(-1, 1) ** (1-c) * cond(s, np.exp(ζ*B)).reshape(-1, 1)**c
    zs = zs / zs.sum(axis=0,keepdims=True)
    # plt.plot(*(basis.T@ zs), '-x',  label="p-q-soft(β=%d;ζ=%d)"%(β,ζ), alpha=0.3)
    plt.scatter(*(basis.T@ zs), alpha=0.3,s=10,c=c, cmap='coolwarm')


plt.scatter(*r @ basis, label='r')
plt.scatter(*s @ basis, label='s')
plt.scatter(*p @ basis, label='p')
plt.scatter(*q @ basis, label='q')
# plt.scatter(* [.67,.33,0,0] @ basis, label='[.68,.32]')


zs.T


# boundary = np.hstack([
#     np.array([1,0,0]).reshape(-1,1)*(1-c) + c * np.array([0,1,0]).reshape(-1,1) ,
#     np.array([0,1,0]).reshape(-1,1)*(1-c) + c * np.array([0,0,1]).reshape(-1,1) ,
#     np.array([0,0,1]).reshape(-1,1)*(1-c) + c * np.array([1,0,0]).reshape(-1,1) 
# ])
# plt.plot(* (basis.T @ boundary), c='k')
# plt.gca().legend()

import itertools
from matplotlib import collections  as mc

verts = np.eye(len(p))
# b = [
#     # u.reshape(-1,1) * (1-c) + c * v.reshape()
#     (u @ basis, v @ basis)
#     for u,v in itertools.combinations(np.eye(len(p)), 2) 
# ]
# b
lc = mc.LineCollection([
    # u.reshape(-1,1) * (1-c) + c * v.reshape()
    (u @ basis, v @ basis)
    for u,v in itertools.combinations(np.eye(len(p)), 2) 
], linewidths=1, colors='k')
ax = plt.gca()
ax.add_collection(lc)
ax.autoscale()
ax.legend()

plt.show()


r
p
cond(zs[:,250],A)
