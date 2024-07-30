import numpy as np
from strongsort_node.conf_kalman_filter import ConfidenceBasedKalmanFilter
import math

class ObjectDescription(): 
    # Future work: replace dist value with probability distribution (ie mean, variance)             
    def __init__(self, frame_id, dist, pitch, yaw, curr_conf, time, 
                 robot_id, descriptor_conf, feature_desc, class_id, 
                 obj_id, children): 
        self.frame_id = frame_id
        self.dist = dist # phi
        self.pitch = pitch # rho
        self.yaw = yaw # theta
        self.curr_conf = curr_conf # latest confidence of detection
        self.time = time # value: stamp.sec + stamp.nanosec
        self.robot_id = robot_id
        self.descriptor_conf = descriptor_conf # Not used
        self.feature_desc = list(feature_desc)
        self.class_id = class_id
        self.obj_id = obj_id
        self.children = children
        
        x = dist * math.sin(pitch) * math.cos(yaw)
        y = dist * math.sin(pitch) * math.sin(yaw)
        z = dist * math.cos(pitch)
        
        self.kalman_filter = ConfidenceBasedKalmanFilter(robot_id, x, y, z, time)
        
    def get_time(self): 
        return self.time
        
    def set_time(self, new_time): 
        self.time = new_time
        