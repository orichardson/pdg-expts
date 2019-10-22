import numpy as np
import scipy.sparse.linalg

# def maxevec_converted(A):
#     n,_ = np.shape(A)
#     z = np.zeros((n, n+1))
#     z[0,0] = 1
# 
#     for i in range(1, n+1): # goes to n+1
#         w = np.max( np.diag(z[:,i-1]) @ A.T, axis=0)
#         z[:,i] = w
#         #z = np.hstack((z,w))
# 
# 
#     z1 = np.diag( 1 / w ) @ z
#     z1 = z1[:, :n]
# 
#     for i in range(n):
#         t = z1[:,i]
#         z1[:,i] = t ** (1/(n-i))
# 
#     mu =  1 / np.min(np.max(z1,axis=1))
#     print("μ=",mu)
# 
#     b = A / mu
# 
#     for i in range(n):
#         w = b[i, :] # row
#         u = b[:, i] # column
#         c = np.outer(u, w) # outer product
# 
#         #print(w.shape,u.shape,c.shape)
# 
#         for j in range(n):
#             for k in range(n):
#                 b[j,k] = max(c[j,k], b[j,k])
# 
#         #b = np.max( [b,c], axis=0)
#     tol = 1E-4
#     for i in range(n):
#         if abs(b[i,i]-1) < tol:
#             evec = b[i,:]
#             return mu,evec    
# 
#     print(b)
#     raise ValueError("No Convergence?")



def maxevec(A, normalize=True):
    n,_ = np.shape(A)
    z = np.zeros((n, n+1))
    z[0,0] = 1
    
    for i in range(1, n+1): # goes to n+1
        w = np.max( np.diag(z[:,i-1]) @ A, axis=0)
        z[:,i] = w
    
    z1 = np.diag( 1 / w ) @ z[:, :n]

    for i in range(n):
        z1[:,i] = z1[:,i] ** (1/(n-i))
    
    mu =  1 / np.min(np.max(z1,axis=1))
    
    b = A.T / mu
    
    for i in range(n):
        b = np.max( [b,np.outer(b[:,i], b[i,:])], axis=0)

    tol = 1E-8
    for i in range(n):
        if abs(b[i,i]-1) < tol:
            evec = b[i,:]
            return mu, evec / np.sum(evec) if normalize else evec
    
    raise ValueError("No Convergence~")
    
def perron(A, normalize=True):
    val, vec = scipy.sparse.linalg.eigs(A, k=1, which='LM')
    vec = np.absolute(vec).flatten()
    return vec / np.sum(vec) if normalize else vec
# Tests
# np.set_printoptions(precision=4, suppress=True)
# 
A = np.array([
    [ 1  , 1/5, 3  , 3   ],
    [ 5  , 1  , 5  , 3   ],
    [ 1/3, 1/5, 1  , 1/5 ],
    [ 1/3, 1/3, 5  , 1   ]])


_,ev = maxevec(A)
ev 
# perron(A)
ev.flatten()
