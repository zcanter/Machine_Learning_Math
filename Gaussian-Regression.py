# Adapted from Coursera's Math for Machine Learning Specialization
import matplotlib.pyplot as plt
import numpy as np

# This is the Gaussian function.
def f (x,mu,sig) :
    return np.exp(-(x-mu)**2/(2*sig**2)) / np.sqrt(2*np.pi) / sig

# Next up, the derivative with respect to μ.
def dfdmu (x,mu,sig) :
    return f(x, mu, sig) * (-1) * (-(x-mu)/(sig**2))

# Finally in this cell, the derivative with respect to σ.
def dfdsig (x,mu,sig) :
    return f (x,mu,sig)*(x-mu)**2/sig**3 + f(x,mu,sig)/(-sig)
    
# The Jacobian: steepest stemp matrix
def steepest_step (x, y, mu, sig, aggression) :
    J = np.array([
        -2*(y - f(x,mu,sig)) @ dfdmu(x,mu,sig),
        -2*(y - f(x,mu,sig)) @ dfdsig(x,mu,sig)
    ])
    step = -J * aggression
    return step
    
# Tests
# First get the heights data, ranges and frequencies
x,y = heights_data()

# Next we'll assign trial values for these.
mu = 155 ; sig = 6
# We'll keep a track of these so we can plot their evolution.
p = np.array([[mu, sig]])

# Plot the histogram for our parameter guess
histogram(f, [mu, sig])
# Do a few rounds of steepest descent.
for i in range(50) :
    dmu, dsig = steepest_step(x, y, mu, sig, 2000)
    mu += dmu
    sig += dsig
    p = np.append(p, [[mu,sig]], axis=0)
# Plot the path through parameter space.
contour(f, p)
# Plot the final histogram.
histogram(f, [mu, sig])
