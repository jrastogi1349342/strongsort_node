import numpy as np
import scipy.linalg
"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    '''6-dimensional state space: x, y, z, vx, vy, vz \n
    Center of object (x, y, z), and their respective velocities, assuming constant 
    velocity.
    '''
    
    # TODO continue
    print("")