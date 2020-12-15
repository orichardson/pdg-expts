import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions()
P = np.array([0.2, 0.3, 0.4])
Q = np.array([0.2, 0.2, 0.3, 0.4, 1])

np.max(P),np.sum(P)
np.max(Q),np.sum(Q)


# BEGIN_FOLD
# %% asdf
def lse1(seq, base=np.e):
    seq = np.array(seq)
    lb = np.log(base)
    return np.log(np.sum(np.exp(seq * lb))) / lb

print( lse1(P)         )
print( lse1(P, 1+1e-8) )
print( lse1(P, np.exp(-100)) )
print( lse1(P, 1e24)   )
print( lse1(P, 1e-24)   )

print( lse1(Q)         )
print( lse1(Q, 1+1e-8) )
print( lse1(Q, 1e24)   )

X = np.linspace(-15, 15, 35)
Y = np.array([ lse1(P, np.exp(x)) for x in X])

plt.plot(X,Y)
# END_FOLD

# %% Let's try to make this happen properly.

# BEGIN_FOLD
def lse2(seq, base=np.e):
    seq = np.array(seq)
    lb = np.log(base)
    return np.log(np.sum(np.exp(seq * lb) - 1) + 1) / lb

print( lse2(P)         ) # unclear  
print( lse2(P, 1+1e-8) ) # ≈ sum    
print( lse2(P, 1e24)   ) # ≈ max    
print( lse2([.5,.1,.3], 1e-5)   ) # ≈ min   

print( lse2(Q)         ) # unclear
print( lse2(Q, 1+1e-8) ) # ≈ sum  
print( lse2(Q, 1e24)   ) # ≈ max  

# But we want it to be the case that the maximum is 1.0 --- 
lse2([lse2(Q, 1.00001), 1],1E-2)

X = np.linspace(-15, 15, 35)
Y = np.array([ lse2(P, np.exp(x)) for x in X])

plt.plot(X,Y)

# END_FOLD


# %%  Unfortunately this does not work

# BEGIN_FOLD
def lse3(seq, base=np.e):
    lb = np.log(base)
    return np.log(np.sum(np.exp(seq * lb)) / len(seq) ) / lb

print( lse3(P)         )
print( lse3(P, 1+1e-8) )
print( lse3(P, 1e24)   )

print( lse3(Q)         )
print( lse3(Q, 1+1e-8) )
print( lse3(Q, 1e24)   )
# END_FOLD
