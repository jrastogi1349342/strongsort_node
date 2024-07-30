import numpy as np
import math
import scipy

class ConfidenceBasedKalmanFilter(): 
    def __init__(self, broker_id, x_init, y_init, z_init, update_time):
        self.broker_id = broker_id
        self.last_updated_time = update_time
        
        self.x = np.array([x_init, y_init, z_init, 0.0, 0.0, 0.0])
        self.P = np.eye(6) * 0.1 # system noise
        self.H = np.eye(3, 6) # update_mat
        self.F = np.eye(6) # motion_mat
        self.Q = np.eye(6) # process noise
        self.R = np.eye(3) # measurement noise
        
    def predict(self, dt): 
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
                
        # Very hacky
        std_pos = [
            self.x[0] / 5.0, 
            self.x[1] / 5.0, 
            self.x[2] / 5.0
        ]
        
        std_vel = [
            self.x[0] / 10.0, 
            self.x[1] / 10.0, 
            self.x[2] / 10.0
        ]
        
        self.Q = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        self.x = np.dot(self.F, self.x)
        self.P = np.linalg.multi_dot((self.F, self.P, self.F.T)) + self.Q
        
    def project(self, confidence=0.0): 
        std_pos = [
            self.x[0] / 20.0, 
            self.x[1] / 20.0, 
            self.x[2] / 20.0
        ]
        
        std_pos = [(1 - confidence) * x for x in std_pos]
        
        self.R = np.diag(np.square(std_pos))
        
        self.x = np.dot(self.H, self.x)
        self.P = np.linalg.multi_dot((self.H, self.P, self.H.T)) + self.R
        
    def project_obj(self, mean, cov, confidence=0.0): 
        std_pos = [
            mean[0] / 20.0, 
            mean[1] / 20.0, 
            mean[2] / 20.0
        ]
        
        std_pos = [(1 - confidence) * x for x in std_pos]
        
        R = np.diag(np.square(std_pos))
        
        new_mean = np.dot(self.H, mean)
        new_cov = np.linalg.multi_dot((self.H, cov, self.H.T)) + R
        
        return new_mean, new_cov
        
    def update(self, measurement, confidence=0.0): 
        orig_mean = self.x
        orig_cov = self.P
        self.project(confidence)
        
        chol_factor, lower = scipy.linalg.cho_factor(self.P, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(orig_cov, self.H.T).T, check_finite=False
        ).T
        
        residual = measurement - self.x
        
        self.x = orig_mean + np.dot(residual, kalman_gain.T)
        self.P = orig_cov - np.linalg.multi_dot((kalman_gain, self.P, kalman_gain.T))
        
    def gating_dist(self, measurement): 
        mean = self.x
        cov = self.P
        
        projected_mean, projected_cov = self.project_obj(mean, cov)
        
        cholesky_factor = np.linalg.cholesky(projected_cov)
        d = measurement - projected_mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, 
                                          lower=True, check_finite=False, 
                                          overwrite_b=True)
        
        print(f"z: {z}")
        
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
    
    # self is distribution 1, inputted mean and cov are distribution 2
    def gaussian_bhattacharyya(self, mean, cov, pose_only): 
        self_mean = self.x
        self_cov = self.P
        
        if pose_only: 
            self_mean, self_cov = self_mean[:3], self_cov[:3, :3]
            mean, cov = mean[:3], cov[:3, :3]

        mean_diff = self_mean - mean

        avg_cov = (self_cov + cov) / 2
        
        first_term = np.linalg.multi_dot((mean_diff, np.linalg.inv(avg_cov), mean_diff.T)) / 8.0
        
        det_avg = np.linalg.det(avg_cov)
        det_one = np.linalg.det(self_cov)
        det_two = np.linalg.det(cov)
        
        sec_term = math.log(det_avg / math.sqrt(det_one * det_two)) / 2
        
        return first_term + sec_term

    def get_state(self): 
        return self.kf.x

    def change_state_broker(self, new_broker_id, x_new, y_new, z_new): 
        self.broker_id = new_broker_id
        self.kf.x = np.array([x_new, y_new, z_new, 0.0, 0.0, 0.0])

# self is distribution 1, inputted mean and cov are distribution 2
def gaussian_bhattacharyya(r0_mean, r0_cov, r1_mean, r1_cov, pose_only): 
    if pose_only: 
        r0_mean, r0_cov = r0_mean[:3], r0_cov[:3, :3]
        r1_mean, r1_cov = r1_mean[:3], r1_cov[:3, :3]

    mean_diff = r0_mean - r1_mean

    avg_cov = (r0_cov + r1_cov) / 2
    
    first_term = np.linalg.multi_dot((mean_diff, np.linalg.inv(avg_cov), mean_diff.T)) / 8.0
    
    det_avg = np.linalg.det(avg_cov)
    det_one = np.linalg.det(r0_cov)
    det_two = np.linalg.det(r1_cov)
    
    sec_term = math.log(det_avg / math.sqrt(det_one * det_two)) / 2
    
    return first_term + sec_term