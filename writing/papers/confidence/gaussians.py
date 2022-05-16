import numpy as np

def toθ(μ, σ2): 
    return np.array([μ/(σ2), -1/(2*σ2)])
    
def invert( θ ):
    return dict(μ= -θ[0] / (2 * θ[1]), σ2= -1 / (2 * θ[1]))


def T(x):
    return np.array([x,x*x])
    

def ET( θ ):
    inv = invert(θ)
    μ = inv['μ']; σ2 = inv['σ2']
    return np.array([μ, μ*μ + σ*σ])

def update(theta0, x, β, nsteps):
    theta = theta0
    for i in range(nsteps):
        # normalize = np.dot(θ, T(x))
        theta += β / nsteps * (T(x) - ET(theta))
        
    return theta


θ = toθ(6, 4); 
update(θ, 6, 0.02, 1000)
invert(θ)
