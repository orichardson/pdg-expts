import numpy as np
from scipy.stats import entropy as entropy

def stoch(X):
    return X / np.sum(X, axis=1)[:, None]

def normalize( A ) :
    return A / np.sum(A)

def dist_to_1link( A ) :
    return stoch(np.array([A]));

shape = (5,)
P_a = np.ones(shape) / np.prod(shape)


p = normalize([1,2,7,3,2])
q = normalize([100,2,7,3,200])



# test -> 
def ent(p, q):
    p_q = normalize(p + q)
    pq = normalize(p * q)
    
    return [ entropy( mid, p) + entropy( mid, q)
            for mid in (p_q, pq)]
