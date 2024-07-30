import numpy as np
from strongsort_node.conf_kalman_filter import ConfidenceBasedKalmanFilter
import math

class ObjectDescription(): 
    # Future work: replace dist value with probability distribution (ie mean, variance)             
    def __init__(self, frame_id, rel_x, rel_y, rel_z, curr_conf, time, 
                 robot_id, descriptor_conf, feature_desc, class_id, 
                 obj_id, children): 
        self.frame_id = frame_id
        self.curr_conf = curr_conf # latest confidence of detection
        self.time = time # value: stamp.sec + stamp.nanosec
        self.robot_id = robot_id
        self.descriptor_conf = descriptor_conf # Not used
        self.feature_desc = list(feature_desc)
        self.class_id = class_id
        self.obj_id = obj_id
        self.children = children
                
        self.kalman_filter = ConfidenceBasedKalmanFilter(robot_id, rel_x, rel_y, rel_z, time)
        
    def get_time(self): 
        return self.time
        
    def set_time(self, new_time): 
        self.time = new_time
        