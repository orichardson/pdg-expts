import numpy as np

a,b,c,d = np.indices([2,2,2,2])

phi1 = 1 + 3*a
phi2 = 7*(a == b)*(b==c)  +  3*(a == c)*(a != b) + 1*(c == b)*(a != b) + 1
phi3 = (a != b)
phi4 = 0.25


f = phi1*phi2*phi3*phi4
Z = np.sum(f, axis=None)

pr = f / Z
