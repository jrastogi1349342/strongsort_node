import numpy as np
from strongsort_node.kalman_filter_mot import KalmanFilter

class ObjectDescription(): 
    # Future work: replace dist value with probability distribution (ie mean, variance)
    def __init__(self):
        self.frame_id = ''
        self.dist = -1 # phi
        self.pitch = -1 # rho
        self.yaw = -1 # theta
        self.time = -1 # value: stamp.sec + stamp.nanosec
        self.robot_id = -1
        self.descriptor_conf = -1
        self.feature_desc = []
        self.class_id = -1
        self.obj_id = -1
        self.children = {}
        self.kalman_filter = KalmanFilter()
        
                
    def __init__(self, frame_id, dist, pitch, yaw, time, robot_id, 
                 descriptor_conf, feature_desc, class_id, obj_id, 
                 children): 
        self.frame_id = frame_id
        self.dist = dist # phi
        self.pitch = pitch # rho
        self.yaw = yaw # theta
        self.time = time # value: stamp.sec + stamp.nanosec
        self.robot_id = robot_id
        self.descriptor_conf = descriptor_conf
        self.feature_desc = list(feature_desc)
        self.class_id = class_id
        self.obj_id = obj_id
        self.children = children
        self.kalman_filter = KalmanFilter()        
        
    def get_time(self): 
        return self.time
        
    def set_time(self, new_time): 
        self.time = new_time
        
    def get_kalman_filter(self): 
        return self.kalman_filter