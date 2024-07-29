import numpy as np
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import random

class ModifiedKalmanFilter(): 
    def __init__(self, broker_id, x_init, y_init, z_init):
        self.dt = -1
        self.broker_id = broker_id
        
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        
        self.kf.x = np.array([x_init, y_init, z_init, 0.0, 0.0, 0.0])
        self.kf.H = np.eye(3, 6) # 3x6 mtx with 1.0 on diagonal, 0.0 elsewhere
        self.kf.P *= 15.0 # Covariance mtx of system noise; default np.eye(6)
        self.kf.R *= 10.0 # Covariance mtx of measurement noise; default np.eye(3)
        # F (state transition mtx) and Q (process noise) changed around at runtime

    def predict(self, dt): 
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt

        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.5, block_size=3, order_by_dim=False)
        
        self.kf.predict()
        
    def update(self, measurement): 
        self.kf.update(measurement)

    def get_state(self): 
        return self.kf.x

    def change_state_broker(self, new_broker_id, x_new, y_new, z_new): 
        self.broker_id = new_broker_id
        self.kf.x = np.array([x_new, y_new, z_new, 0.0, 0.0, 0.0])
