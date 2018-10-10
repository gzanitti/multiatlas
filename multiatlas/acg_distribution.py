from __future__ import division

from warnings import warn
import numpy as np
from scipy import special
from dipy.data import get_sphere

def density(points, Lambda):
    """Density function of the Angular Central Gaussian Distribution"""
    q = Lambda.shape[0]
    a_inv =  special.gamma(q / 2.) / (2. * np.pi ** (q / 2.))
    det_inv = np.linalg.det(Lambda) ** -.5
    xpx = (np.linalg.solve(Lambda, points.T) * points.T).sum(0)
    return a_inv * det_inv * xpx ** -(q / 2.)


def lambda_estimator(points, iters=1000, epsilon=1e-8):
    """Estimator of the parameter for the Angular Central Gaussian
       Distribution according to Statistical Analysis for the Angular
       Central Gaussian Distribution on the Sphere David E. Tyler,
       Biometrika, Vol. 74, No. 3 (Sep., 1987), pp. 579-589 Eq. 3."""
    q = points.shape[1]
    Lambda = np.eye(q)
    points_squared = points[:,:,None] * points[:,None,:]

    for i in range(iters):
        if np.linalg.det(Lambda) == 0:
            M = np.eye(q) - Lambda
            M /= np.linalg.norm(M)
            Lambda = Lambda + M * epsilon
        denominator = (
            points * np.linalg.solve(Lambda, points.T).T
        ).sum(1)
        denominator = denominator[:, None, None]
        Lambda_new = q * (
            (points_squared / denominator).sum(0) / (1. / denominator).sum(0)
        )

        if np.linalg.norm(Lambda - Lambda_new) < epsilon:
            break
        Lambda = Lambda_new
    else:
        warn('Convergence not achieved, stopping by number of iterations.')
    return Lambda / np.linalg.norm(Lambda)
