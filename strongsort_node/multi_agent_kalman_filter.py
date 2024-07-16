import numpy as np

class KalmanFilter(): 
    def __init__(self):
        self.ref_frame = -1
        self.update_num = -1
        self.x_vec = np.array([[0], [0], [0], [0], [0], [0]])
        self.P = np.eye(6, dtype=int) * 15
        # TODO Assign dt part to A dynamically
        self.A = np.eye(6, dtype=int)
        self.H = np.array([[1, 1, 1, 0, 0, 0]]) # Transpose is self.H.T
        self.R = np.eye(3, dtype=int) * 5
        self.Q = np.array([[1, 0, 0, 0, 0, 0], 
                           [0, 1, 0, 0, 0, 0], 
                           [0, 0, 1, 0, 0, 0], 
                           [0, 0, 0, 5, 2, 2], 
                           [0, 0, 0, 2, 5, 2], 
                           [0, 0, 0, 2, 2, 5]])

    def get_update_num(self): 
        return self.update_num
        
    def set_update_num(self, new_num): 
        self.update_num = new_num
        
    def set_A(self, dt): 
        self.A[0, 3] = dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt


# import scipy.linalg
# """
# Table for the 0.95 quantile of the chi-square distribution with N degrees of
# freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
# function and used as Mahalanobis gating threshold.
# """
# chi2inv95 = {
#     1: 3.8415,
#     2: 5.9915,
#     3: 7.8147,
#     4: 9.4877,
#     5: 11.070,
#     6: 12.592,
#     7: 14.067,
#     8: 15.507,
#     9: 16.919}


# class KalmanFilter(object):
#     '''6-dimensional state space: phi, rho, theta, v_phi, v_rho, v_theta \n
#     Center of object (phi, rho, theta), where phi is median depth, 
#     and their respective velocities, assuming constant velocity.
#     '''
    
#     def __init__(self):
#         ndim, dt = 3, 1
        
#         # Create Kalman filter model matrices.
#         self._motion_mat = np.eye(2 * ndim, 2 * ndim)
#         for i in range(ndim):
#             self._motion_mat[i, ndim + i] = dt

#         self._update_mat = np.eye(ndim, 2 * ndim)

#         # Motion and observation uncertainty are chosen relative to the current
#         # state estimate. These weights control the amount of uncertainty in
#         # the model. This is a bit hacky.
#         # TODO figure this out
#         self._std_weight_position = 1. / 20
#         self._std_weight_velocity = 1. / 160

#     def initiate(self, measurement):
#         """Create track from unassociated measurement.
#         Parameters
#         ----------
#         measurement : ndarray
#             Bounding box coordinates (phi, rho, theta)
#         Returns
#         -------
#         (ndarray, ndarray)
#             Returns the mean vector (6 dimensional) and covariance matrix (6x6
#             dimensional) of the new track. Unobserved velocities are initialized
#             to 0 mean.
#         """
#         mean_pos = measurement
#         mean_vel = np.zeros_like(mean_pos)
#         mean = np.r_[mean_pos, mean_vel]

#         # Multiplication of 2 and 10 is hacky
#         std = [
#             2 * self._std_weight_position * measurement[0],   # the median distance
#             2 * self._std_weight_position * measurement[1],   # the center point rho
#             2 * self._std_weight_position * measurement[2],   # the center point theta
#             10 * self._std_weight_velocity * measurement[0],
#             10 * self._std_weight_velocity * measurement[1],
#             10 * self._std_weight_velocity * measurement[2]]
#         covariance = np.diag(np.square(std))
#         return mean, covariance

#     def predict(self, mean, covariance):
#         """Run Kalman filter prediction step.
#         Parameters
#         ----------
#         mean : ndarray
#             The 6 dimensional mean vector of the object state at the previous
#             time step.
#         covariance : ndarray
#             The 6x6 dimensional covariance matrix of the object state at the
#             previous time step.
#         Returns
#         -------
#         (ndarray, ndarray)
#             Returns the mean vector and covariance matrix of the predicted
#             state. Unobserved velocities are initialized to 0 mean.
#         """
#         std_pos = [
#             self._std_weight_position * mean[0],
#             self._std_weight_position * mean[1],
#             1 * mean[2],
#             self._std_weight_position * mean[3]]
#         std_vel = [
#             self._std_weight_velocity * mean[0],
#             self._std_weight_velocity * mean[1],
#             0.1 * mean[2],
#             self._std_weight_velocity * mean[3]]
#         motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

#         mean = np.dot(self._motion_mat, mean)
#         covariance = np.linalg.multi_dot((
#             self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

#         return mean, covariance

#     def project(self, mean, covariance, confidence=.0):
#         """Project state distribution to measurement space.
#         Parameters
#         ----------
#         mean : ndarray
#             The state's mean vector (8 dimensional array).
#         covariance : ndarray
#             The state's covariance matrix (8x8 dimensional).
#         confidence: (dyh) 检测框置信度
#         Returns
#         -------
#         (ndarray, ndarray)
#             Returns the projected mean and covariance matrix of the given state
#             estimate.
#         """
#         std = [
#             self._std_weight_position * mean[3],
#             self._std_weight_position * mean[3],
#             1e-1,
#             self._std_weight_position * mean[3]]


#         std = [(1 - confidence) * x for x in std]

#         innovation_cov = np.diag(np.square(std))

#         mean = np.dot(self._update_mat, mean)
#         covariance = np.linalg.multi_dot((
#             self._update_mat, covariance, self._update_mat.T))
#         return mean, covariance + innovation_cov

#     def update(self, mean, covariance, measurement, confidence=.0):
#         """Run Kalman filter correction step.
#         Parameters
#         ----------
#         mean : ndarray
#             The predicted state's mean vector (8 dimensional).
#         covariance : ndarray
#             The state's covariance matrix (8x8 dimensional).
#         measurement : ndarray
#             The 4 dimensional measurement vector (x, y, a, h), where (x, y)
#             is the center position, a the aspect ratio, and h the height of the
#             bounding box.
#         confidence: (dyh)检测框置信度
#         Returns
#         -------
#         (ndarray, ndarray)
#             Returns the measurement-corrected state distribution.
#         """
#         projected_mean, projected_cov = self.project(mean, covariance, confidence)

#         chol_factor, lower = scipy.linalg.cho_factor(
#             projected_cov, lower=True, check_finite=False)
#         kalman_gain = scipy.linalg.cho_solve(
#             (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
#             check_finite=False).T
#         innovation = measurement - projected_mean

#         new_mean = mean + np.dot(innovation, kalman_gain.T)
#         new_covariance = covariance - np.linalg.multi_dot((
#             kalman_gain, projected_cov, kalman_gain.T))
#         return new_mean, new_covariance

#     def gating_distance(self, mean, covariance, measurements,
#                         only_position=False):
#         """Compute gating distance between state distribution and measurements.
#         A suitable distance threshold can be obtained from `chi2inv95`. If
#         `only_position` is False, the chi-square distribution has 4 degrees of
#         freedom, otherwise 2.
#         Parameters
#         ----------
#         mean : ndarray
#             Mean vector over the state distribution (8 dimensional).
#         covariance : ndarray
#             Covariance of the state distribution (8x8 dimensional).
#         measurements : ndarray
#             An Nx4 dimensional matrix of N measurements, each in
#             format (x, y, a, h) where (x, y) is the bounding box center
#             position, a the aspect ratio, and h the height.
#         only_position : Optional[bool]
#             If True, distance computation is done with respect to the bounding
#             box center position only.
#         Returns
#         -------
#         ndarray
#             Returns an array of length N, where the i-th element contains the
#             squared Mahalanobis distance between (mean, covariance) and
#             `measurements[i]`.
#         """
#         mean, covariance = self.project(mean, covariance)

#         if only_position:
#             mean, covariance = mean[:2], covariance[:2, :2]
#             measurements = measurements[:, :2]

#         cholesky_factor = np.linalg.cholesky(covariance)
#         d = measurements - mean
#         z = scipy.linalg.solve_triangular(
#             cholesky_factor, d.T, lower=True, check_finite=False,
#             overwrite_b=True)
#         squared_maha = np.sum(z * z, axis=0)
#         return squared_maha
    
    
# ----------------- Online implementation that may be more simple to understand --------------
# import numpy as np

# class KalmanFilter(object):
#     def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

#         if(F is None or H is None):
#             raise ValueError("Set proper system dynamics.")

#         self.n = F.shape[1]
#         self.m = H.shape[1]

#         self.F = F
#         self.H = H
#         self.B = 0 if B is None else B
#         self.Q = np.eye(self.n) if Q is None else Q
#         self.R = np.eye(self.n) if R is None else R
#         self.P = np.eye(self.n) if P is None else P
#         self.x = np.zeros((self.n, 1)) if x0 is None else x0

#     def predict(self, u = 0):
#         self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
#         self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
#         return self.x

#     def update(self, z):
#         y = z - np.dot(self.H, self.x)
#         S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
#         K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
#         self.x = self.x + np.dot(K, y)
#         I = np.eye(self.n)
#         self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
#         	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

# def example():
# 	dt = 1.0/60
# 	F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
# 	H = np.array([1, 0, 0]).reshape(1, 3)
# 	Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
# 	R = np.array([0.5]).reshape(1, 1)

# 	x = np.linspace(-10, 10, 100)
# 	measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, 2, 100)

# 	kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
# 	predictions = []

# 	for z in measurements:
# 		predictions.append(np.dot(H,  kf.predict())[0])
# 		kf.update(z)

# 	import matplotlib.pyplot as plt
# 	plt.plot(range(len(measurements)), measurements, label = 'Measurements')
# 	plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
# 	plt.legend()
# 	plt.show()

# if __name__ == '__main__':
#     example()