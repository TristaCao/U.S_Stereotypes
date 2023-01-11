from __future__ import division
import numpy as np
from numpy.linalg import norm

def column_squishing(z, do_proj=True):
    # input: z has z_2 >= z_3 >= z_4 >= ... >= z_n
    # returns the projection of z into { x in R : 0 <= x_i <= x_1 <= 1 }
    # this is algorithm 5 from:
    #   Factoring nonnegative matrices with linear programs
    #   by Bittorf et al., June 2012
    #   http://pages.cs.wisc.edu/~brecht/papers/12.Bit.EtAl.HOTT.pdf
    proj01 = (lambda a: max(0, min(1, a))) if do_proj else (lambda a: a)
    proj0_ = (lambda a: max(0, a)) if do_proj else (lambda a: a)
    n = z.shape[0]
    assert len(z.shape) == 1
    assert all([z[i] >= z[i+1] for i in range(1, n-1)])
    mu = z[0]
    kc = n-1
    for k in range(1, n):
        if z[k] <= proj01(mu):
            kc = k - 1
            break
        mu = mu * k / (k+1) + z[k] / (k+1)
    x = np.zeros(n) + proj01(mu)
    for k in range(kc+1, n):
        x[k] = proj0_(z[k])
    return x

def min_delta(C, j):
    # solve:
    #   min_delta sum_i delta_i^2 st delta_j >= delta_i + C_i for i != j
    # do a change of variables where
    #   z = delta + D
    # then we want to solve
    #   min_x ||x-z|| st x_j >= x_i for i != j
    # after reordering C so that D[0] = C[j] and D[1:] is sorted(C[!j])
    # and then need to un-sort the results
    order = (-C).argsort()
    j_idx = (order == j).nonzero()[0][0]
    order2 = np.concatenate([[j], order[:j_idx], order[j_idx+1:]])
    proj = column_squishing(C[order2], False)
    return proj[order2.argsort()] - C

def multiclass_update(A, x, j):
    # given matrix A (R^k*d), x (R^d) and j, find B that solves:
    #   min_B ||B-A||^2 st (xB)_j >= (xB)_i + 1 for all i != j
    # observe that any change will be in the direction of x
    # so compute scalars:
    #   C_i = [ a_i - a_j + 1 ] / ||x||^2
    # where a_i is x*A[i,:]
    k, d = A.shape
    a = A.dot(x)
    C = (a - a[j] + 1) / x.dot(x)
    C[j] = 0
    delta = min_delta(C, j)
    # print(delta.dot(delta))
    return A + delta.reshape((k,1)).dot(x.reshape(1,d))
