#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
import math
import heapq

from geometry_msgs.msg import Vector3, Quaternion, Transform
from scipy.spatial.transform import Rotation as R

from strongsort_msgs.msg import MOTGlobalDescriptor, MOTGlobalDescriptors
from strongsort_node.neighbors_manager import NeighborManager
from strongsort_node.disjoint_set_associations import DisjointSetAssociations

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

class MOTDescriptorGraphConstruction(object): 
    def __init__(self, params, node):
        self.params = params
        self.node = node
        self.neighbor_manager = NeighborManager(self.node, self.params)

        self.unified = DisjointSetAssociations()

        # self.tf_buffer = Buffer()
        
        # Contains MOTGlobalDescriptor messages from all neighboring agents
        # Should have all detections that neighbors found because only the unseen information
        # is sent here
        # Overall: dictionary of detections with the stamp as key and dictionary of robot_id's as the value
        #          Each robot_id corresponds to an array of MOTGlobalDescriptor objects
        self.all_neighbors_info = {}
        for i in range(self.params['max_nb_robots']): 
            # Collectively decrease confidence in object location at once for old detections
            self.all_neighbors_info[i] = {}

        # Get info from all robots --> fix this
        self.info_sub = self.node.create_subscription(
            MOTGlobalDescriptors, 
            f"/{self.params['name_space']}/mot", 
            self.info_callback, 
            100)


        # TODO insert new information into dictionary, ordering by time
        # After that, choose a broker, based on neighboring agents present (using cslam's 
        # neighbors_monitor.py and neighbor_monitor.py)
        # Add 'unified object id' to some message somewhere
        # Do clustering, using check_same_location
        # Use tracking concepts to decrease confidence location for old detections
    def info_callback(self, global_desc_msg): 
        highest_prev_stamp = -1.0
        new_stamp = global_desc_msg.header.stamp.sec + (global_desc_msg.header.stamp.nanosec / 1000000000)
        
        if highest_prev_stamp == -1.0: 
            self.all_neighbors_info[new_stamp] = {}
            self.all_neighbors_info[new_stamp][global_desc_msg.descriptors[0].robot_id] = global_desc_msg.descriptors
            highest_prev_stamp = new_stamp
        else: 
            if new_stamp - highest_prev_stamp < 0.010: # 10 ms, messages from close enough of a time
                stamp_dict = self.all_neighbors_info[highest_prev_stamp]
                if global_desc_msg.descriptors[0].robot_id in stamp_dict: 
                    stamp_dict[global_desc_msg.descriptors[0].robot_id].append(global_desc_msg.descriptors)
                else: 
                    stamp_dict[global_desc_msg.descriptors[0].robot_id] = global_desc_msg.descriptors
            else: # Too long of a time between detection sets
                stamp_dict = self.all_neighbors_info[new_stamp]

                if global_desc_msg.descriptors[0].robot_id in stamp_dict: 
                    stamp_dict[global_desc_msg.descriptors[0].robot_id].append(global_desc_msg.descriptors)
                else: 
                    stamp_dict[global_desc_msg.descriptors[0].robot_id] = global_desc_msg.descriptors

                highest_prev_stamp = new_stamp
        
        # # Each MOTGlobalDescriptors message only comes from one robot_id
        # obj_class_dict = self.all_neighbors_info[global_desc_msg.descriptors[0].robot_id]
        
        # for info in global_desc_msg.descriptors: # info is of type MOTGlobalDescriptor
        #     obj_class_dict[stamp].append(info)
            

    def detect_inter_robot_associations(self): 
        '''Detect inter-robot object associations
        '''
        # neighbors_is_in_range: bool --> in range of any others?
        neighbors_is_in_range, neighbors_in_range_list = self.neighbor_manager.check_neighbors_in_range()
        
        if len(neighbors_in_range_list) > 0 and self.neighbor_manager.local_robot_is_broker():
            # Messages grouped together by approximate time chunks --> a bunch of MOTGlobalDescriptor topics
            # Do hierarchical clustering here
            last_dets = list(self.all_neighbors_info.values())[-1]
            self.cluster(last_dets, neighbors_in_range_list, 0.6, 1.0)

            # TODO get info every frame, and generalize StrongSORT + OSNet pipeline to relative poses from 
            # the perspective of the robot with the lowest agent ID (modifying Kalman Filter implementation) 
            # to deal with multi-agent associations

            # Apply some technique to send the resulting clusters out


    def cluster(self, dets, curr_neighbors_in_range_list, compact_desc_min_sim, location_epsilon):
        '''Takes in all detections from a certain timestamp
        - dets: dict with robot_id as key, array of MOTGlobalDescriptor objects as value
        - curr_neighbors_in_range_list: list of robot_id's, which is monotonically increasing
        - compact_desc_min_sim: float minimum similarity value between two compact descriptors
        - location_epsilon: float minimum similarity between two locations
        ''' 
        
        # curr_neighbors_in_range_list is array of robot_id's
        self.unified.insert_arr([f'{x.robot_id}.{x.obj_id}' for x in dets[curr_neighbors_in_range_list[0]]])
        
        # for i in curr_neighbors_in_range_list[0]: # i is MOTGlobalDescriptor object
        #     key = f"{str(i.robot_id)}.{str(i.obj_id)}"
        #     unified[key] = DisjointSetAssociations(len(curr_neighbors_in_range_list[0]))
        
        for i in range(1, len(curr_neighbors_in_range_list) - 1): 
            first = curr_neighbors_in_range_list[i]
            second = curr_neighbors_in_range_list[i + 1]
            
            self.unified.insert_arr([f'{x.robot_id}.{x.obj_id}' for x in dets[second]])
            
            for j in dets[first]: # j, k are MOTGlobalDescriptor objects
                heap = []
                
                for k in dets[second]: 
                    # Assuming each class is very different from each other class, don't cluster if detected as a 
                    # different class
                    if j.obj_class_id != k.obj_class_id: 
                        continue

                    # # Triangle approach
                    # check_odom = self.check_same_location(
                    #     j.distance, 
                    #     [j.angle_x, j.angle_y, j.angle_z], 
                    #     k.distance, 
                    #     [k.angle_x, k.angle_y, k.angle_z], 
                    #     k.colocalization)
                    
                    # r0_dist, r0_pitch, r0_yaw, r1_dist, r1_pitch, r1_yaw, colocalization, epsilon
                    check_odom = self.check_same_location(j.distance, j.pitch, j.yaw, k.distance, 
                                                          k.pitch, k.yaw, j.colocalization, location_epsilon)
                    
                    if not check_odom: 
                        continue
                    
                    # Assume there's only one bounding box around any given object, due to NMS working perfectly
                    check_feature_desc = self.check_features(j.best_descriptor, k.best_descriptor)
                    
                    if check_feature_desc > compact_desc_min_sim: 
                        heapq.heappush(heap, (-1 * check_feature_desc, k.obj_id)) # heapq is a min heap, NOT a max heap
            
                if len(heap) != 0: 
                    _, closest_obj_id = heapq.heappop(heap)                
                    self.unified.union(f"{j.robot_id}.{j.obj_id}", f"{k.robot_id}.{closest_obj_id}")
                        
                        


    # # This attempt utilizes triangle inequality + Law of Cosine/Law of Sine
    # def check_same_location(self, r0_dist, r0_angles, r1_dist, r1_angles, colocalization, epsilon): 
    #     '''Check if two distance estimates of an object intuitively make sense
    #     Arguments:
    #     - r0_dist: float32 distance from r0 to object
    #     - r0_angles: np.array([x, y, z]) angles from r0 to object
    #     - r1_dist: float32 distance from r1 to object
    #     - r1_angles: np.array([x, y, z]) angles from r1 to object
    #     - colocalization: geometry_msgs/Pose message for relative location of r1 from the 
    #     reference frame of r0
    #     '''
        
    #     # # Check if it obeys the triangle inequality
    #     # r0_to_r1_dist = math.sqrt(colocalization.position.x ** 2 + colocalization.position.y ** 2 + colocalization.position.z ** 2)
    #     # if r0_dist + r0_to_r1_dist <= r1_dist or r1_dist + r0_to_r1_dist <= r0_dist or r0_dist + r1_dist <= r0_to_r1_dist: 
    #     #     return False
        
    #     # ...
        
    def check_same_location(self, r0_dist, r0_pitch, r0_yaw, r1_dist, r1_pitch, r1_yaw, colocalization, epsilon): 
        # Assumes user is looking forward
        # Using spherical coords: pitch is 0 (up) to 180 (down) degrees, yaw is 0 to 360 degrees
        r0_base = [r0_dist * math.cos(r0_pitch) * math.sin(r0_yaw), r0_dist * math.cos(r0_pitch) * math.cos(r0_yaw), r0_dist * math.cos(r0_pitch)]
        r1_base = [r1_dist * math.cos(r1_pitch) * math.sin(r1_yaw), r1_dist * math.cos(r1_pitch) * math.cos(r1_yaw), r1_dist * math.cos(r1_pitch)]
        
        r1_trans = []
        r1_trans[0] = r1_base[0] - colocalization.transform.translation.x
        r1_trans[1] = r1_base[1] - colocalization.transform.translation.y
        r1_trans[2] = r1_base[2] - colocalization.transform.translation.z

        quat_ros = colocalization.transform.rotation
        quat = R.from_quat([quat_ros.x, quat_ros.y, quat_ros.z, quat_ros.w])
        
        r1_trans_rot = quat.as_matrix * r1_trans
        
        return distance.euclidean(r0_base, r1_trans_rot) < epsilon
        
        
    # TODO write comments
    def check_features(self, r0_descriptor, r1_descriptor): 
        '''
        '''
        return cosine_similarity(r0_descriptor, r1_descriptor)
    