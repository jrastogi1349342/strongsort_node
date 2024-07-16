#!/usr/bin/env python3
# Loop Closure Detection service
# Abstraction to support multiple implementations of loop closure detection for benchmarking

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from sensor_msgs.msg import Image

from strongsort_node.associations import ObjectAssociation


class AssociationsROSDriver(Node):
    """ Global image descriptor matching for loop closure detection """

    def __init__(self):
        """Initialization and parameter parsing"""
        super().__init__('loop_closure_detection')

        self.declare_parameters(
            namespace='',
            parameters=[('robot_id', 0), 
                        ('max_nb_robots', 3),
                        ('sort.max_occlusion_time', 20), # seconds
                        ('sort.compact_desc_min_similarity', 0.6), # minimum similarity value between two compact descriptors
                        ('sort.location_epsilon', 0.8), # maximum difference between two locations 
                        ('sort.re_cluster_secs', 3.0), # number of seconds to wait before re-clustering associations
                        ])
        self.params = {}
        
        self.params['robot_id'] = self.get_parameter('robot_id').value
        self.params['max_nb_robots'] = self.get_parameter('max_nb_robots').value
        self.params['sort.max_occlusion_time'] = self.get_parameter(
            'sort.max_occlusion_time').value
        self.params['sort.compact_desc_min_similarity'] = self.get_parameter(
            'sort.compact_desc_min_similarity').value
        self.params['sort.location_epsilon'] = self.get_parameter(
            'sort.location_epsilon').value

        self.obj_association_driver = ObjectAssociation(
            self.params, self)
        
        self.inter_robot_detection_timer = self.create_timer(
            self.params['sort.re_cluster_secs'],
            self.obj_association_driver.detect_inter_robot_associations, 
            clock=Clock())

        # self.apply_kf_timer = self.create_timer(
        #     self.params['sort.kalman_filter_secs'], 
        #     self.obj_association_driver.apply_kf, 
        #     clock=Clock()
        # )

if __name__ == '__main__':
    rclpy.init(args=None)
    driver = AssociationsROSDriver()
    driver.get_logger().info('Initialization for multi-agent associations done.')
    rclpy.spin(driver)
    rclpy.shutdown()
