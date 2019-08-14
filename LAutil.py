
# Written by Zac Canter (many adapted from Coursera's Mathematics from Machine Learning: Linear Algebra)

import numpy as np
import numpy.linalg as la

# Outputs orthonormal basis for some matrix using Gramm-Schmitt
def gsBasis(A) :
    B = np.array(A, dtype=np.float_) 
    for i in range(B.shape[1]) :
        for j in range(i) :
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]
        if la.norm(B[:, i]) > 1e-14:
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else :
            B[:, i] = np.zeros_like(B[:, i])
    return B

# Outputs dimensions of an orthonormal basis using Gramm-Schmitt
def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))

# Outputs matrix reflected about some orthonormal basis vectors also uses Gramm-Schmitt
def refection_matrix(A):
    # get basis via Gramm Schmitt
    E = gsBasis(A)
    # desired reflection (about y-axis)
    T =  np.array([[1, 0], [0, -1]])
    # formula to output
    R = E@T@transpose(E)
    return R

