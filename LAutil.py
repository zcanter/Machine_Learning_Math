
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

# PageRank Algorithm (linkMatrix is the initial probabilities of page network, d is dampening factor -> good to try d=0.5)
def pageRank(linkMatrix, d):
    n = linkMatrix.shape[0]
    M = d * linkMatrix + (1-d)/n * np.ones([n, n])
    r = 100*np.ones(n)/n
    lastR = r
    r = M @ r
    i = 0
    while la.norm(lastR - r) > 0.01 :
        lastR = r
        r = M @ r
        i += 1
    print(str(i) + " iterations to convergence.")
    return r

# Compute the angle using the inner product
A = np.array([[2,-1], [-1,4]])
x = [1,1]
y = [0,2]

def angle(A, x, y):
    """Computes the angle in radians"""
    x,y,A = map(np.array, [x,y,A])
    norm_x = np.sqrt(x.T @ A @ x)
    norm_y = np.sqrt(y.T @ A @ y)
    sin = (x.T @ A @ y) / (norm_x * norm_y)
    # inverse sin
    return np.arccos(sin)

a = angle(A,x,y)
print(a)

# Projections

# Projection 1d

def projection_matrix_1d(b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        b: ndarray of dimension (D, 1), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    P = np.dot(b,b.T)/np.linalg.norm(b)**2
    return P

def project_1d(x, b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        x: the vector to be projected
        b: ndarray of dimension (D, 1), the basis for the subspace
    
    Returns:
        y: ndarray of shape (D, 1) projection of x in space spanned by b
    """
    return projection_matrix_1d(b)@x

# Projection onto a general (higher-dimensional) subspace
def projection_matrix_general(B):
    """Compute the projection matrix onto the space spanned by the columns of `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    P = B@np.linalg.inv(B.T@B)@B.T
    return P

def project_general(x, B):
    """Compute the projection matrix onto the space spanned by the columns of `B`
    Args:
        x: ndarray of dimension (D, 1), the vector to be projected
        B: ndarray of dimension (D, M), the basis for the subspace
    
    Returns:
        p: projection of x onto the subspac spanned by the columns of B; size (D, 1)
    """
    return projection_matrix_general(B) @ x
