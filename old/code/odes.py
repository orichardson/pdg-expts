def dSym(A):
    return - (A + A.T)/2
    
def dTrans(A):
    return np.max( A[:, None] * A[None, :] )
