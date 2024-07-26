import numpy as np
from strongsort_node.kalman_filter_mot import ModifiedKalmanFilter
import math

class ObjectDescription(): 
    # Future work: replace dist value with probability distribution (ie mean, variance)             
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
        
        x = dist * math.sin(pitch) * math.cos(yaw)
        y = dist * math.sin(pitch) * math.sin(yaw)
        z = dist * math.cos(pitch)
        
        self.kalman_filter = ModifiedKalmanFilter(robot_id, x, y, z)
        
    def get_time(self): 
        return self.time
        
    def set_time(self, new_time): 
        self.time = new_time
        