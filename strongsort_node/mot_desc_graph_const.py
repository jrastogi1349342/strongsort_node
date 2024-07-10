#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
import math

from strongsort_node.cosplace import CosPlace
from strongsort_msgs.msg import CroppedObject, CroppedObjectArray, KeyframeOdomRGB, MOTGlobalDescriptor, MOTGlobalDescriptors

class MOTDescriptorGraphConstruction(object): 
    def __init__(self, params, node):
        self.params = params
        self.node = node

        # Get info from all robots
        self.info_sub = self.node.create_subscription(
            MOTGlobalDescriptors, 
            f"/{self.params['name_space']}{self.params['video_topic']}/mot", 
            self.info_callback, 
            100)
        
        # TODO think about this
        
        # TODO insert neighbors_monitor.py and neighbor_monitor.py in this repo
        
    def info_callback(self, global_desc_msg): 


        print("at the hard part :/")
        
        
    def check_same_location(self, r0_dist, r0_angles, r1_dist, r1_angles, colocalization): 
        '''
        Arguments:
        - r0_dist: float32 distance from r0 to object
        - r0_angles: np.array([x, y, z]) angles from r0 to object
        - r1_dist: float32 distance from r1 to object
        - r1_angles: np.array([x, y, z]) angles from r1 to object
        - colocalization: geometry_msgs/Pose message for relative location of r1 from the 
        reference frame of r0
        '''
        
        # Check if it obeys the triangle inequality
        r0_to_r1_dist = math.sqrt(colocalization.position.x ** 2 + colocalization.position.y ** 2 + colocalization.position.z ** 2)
        if r0_dist + r0_to_r1_dist < r1_dist or r1_dist + r0_to_r1_dist < r0_dist or r0_dist + r1_dist < r0_to_r1_dist: 
            return False
        
        # Assuming z is constant: 
        
        
        print("here we go")