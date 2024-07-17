#!/usr/bin/env python3
# Loop Closure Detection service
# Abstraction to support multiple implementations of loop closure detection for benchmarking

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from std_msgs.msg import UInt32

from strongsort_node.associations import ObjectAssociation


class AssociationsROSDriver(Node):
    """ Global image descriptor matching for loop closure detection """

    def __init__(self):
        """Initialization and parameter parsing"""
        super().__init__('loop_closure_detection')

        self.declare_parameters(
            namespace='',
            parameters=[('name_space', "hl2"), 
                        ('robot_id', 0), 
                        ('max_nb_robots', 3),
                        ('sort.max_occlusion_time', 20), # seconds
                        ('sort.compact_desc_min_similarity', 0.6), # minimum similarity value between two compact descriptors
                        ('sort.location_epsilon', 0.8), # maximum difference between two locations 
                        ('sort.re_cluster_secs', 8.0), # number of seconds to wait before re-clustering associations
                        ('neighbor_management.enable_neighbor_monitoring', True),
                        ('neighbor_management.init_delay_sec', 5.0), 
                        ('neighbor_management.max_heartbeat_delay_sec', 5.0), 
                        ('neighbor_management.heartbeat_period_sec', 1.0)
                        ])
        self.params = {}
        
        self.params['name_space'] = self.get_parameter('name_space').value
        self.params['robot_id'] = self.get_parameter('robot_id').value
        self.params['max_nb_robots'] = self.get_parameter('max_nb_robots').value
        self.params['sort.max_occlusion_time'] = self.get_parameter(
            'sort.max_occlusion_time').value
        self.params['sort.compact_desc_min_similarity'] = self.get_parameter(
            'sort.compact_desc_min_similarity').value
        self.params['sort.location_epsilon'] = self.get_parameter(
            'sort.location_epsilon').value
        self.params['sort.re_cluster_secs'] = self.get_parameter(
            'sort.re_cluster_secs').value
        self.params['neighbor_management.enable_neighbor_monitoring'] = self.get_parameter(
            'neighbor_management.enable_neighbor_monitoring').value
        self.params['neighbor_management.init_delay_sec'] = self.get_parameter(
            'neighbor_management.init_delay_sec').value
        self.params['neighbor_management.max_heartbeat_delay_sec'] = self.get_parameter(
            'neighbor_management.max_heartbeat_delay_sec').value
        self.params['neighbor_management.heartbeat_period_sec'] = self.get_parameter(
            'neighbor_management.heartbeat_period_sec').value

        self.obj_association_driver = ObjectAssociation(
            self.params, self)
        
        self.inter_robot_detection_timer = self.create_timer(
            self.params['sort.re_cluster_secs'],
            self.obj_association_driver.detect_inter_robot_associations, 
            clock=Clock())
        
        self.heartbeat_publisher = self.create_publisher(UInt32, f"/r{self.params['robot_id']}/mot/heartbeat", 100)
        self.heartbeat_timer = self.create_timer(
            self.params['neighbor_management.heartbeat_period_sec'], 
            self.send_heartbeat, 
            clock=Clock())

        # self.apply_kf_timer = self.create_timer(
        #     self.params['sort.kalman_filter_secs'], 
        #     self.obj_association_driver.apply_kf, 
        #     clock=Clock()
        # )
        
    def send_heartbeat(self): 
        self.heartbeat_publisher.publish(UInt32(data=self.params['robot_id']))

if __name__ == '__main__':
    rclpy.init(args=None)
    driver = AssociationsROSDriver()
    driver.get_logger().info('Initialization for multi-agent associations done.')
    rclpy.spin(driver)
    rclpy.shutdown()
