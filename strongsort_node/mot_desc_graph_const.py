#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

from strongsort_node.cosplace import CosPlace
from strongsort_msgs.msg import CroppedObject, CroppedObjectArray, KeyframeOdomRGB, MOTGlobalDescriptor, MOTGlobalDescriptors

class MOTDescriptorGraphConstruction(object): 
    def __init__(self):
        """
        Initialization
        """

        # Get info from all robots
        self.info_sub = self.node.create_subscription(
            MOTGlobalDescriptors, 
            '/hl2/rm_vlc_leftfront/image/mot', 
            self.info_callback, 
            100)
        
        # TODO think about this
        
        