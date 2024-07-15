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

class ObjectAssociation(object): 
    def __init__(self, params, node):
        self.params = params
        self.node = node
        self.neighbor_manager = NeighborManager(self.node, self.params)

        self.unified = DisjointSetAssociations(self.params['max_nb_robots'])
        self.latest_time_entered = -1 # last time new information entered via callback
        self.last_time_clustered = -1 # last time the information was clustered

        # Get info from all robots --> TODO fix this
        self.info_sub = self.node.create_subscription(
            MOTGlobalDescriptors, 
            f"/{self.params['name_space']}/mot", 
            self.info_callback, 
            100)


    # Version 2
    # Update last seen time of all info in unified set, and generalize old location with EKF
    # Insert all new messages into unified disjoint set, if new; if old, update last seen time to curr
    # For each new message: add to cluster
    # If object has not been seen in [some amount of time], delete
    def info_callback(self, global_desc_msg): 
        curr_time = global_desc_msg.header.stamp.sec + (global_desc_msg.header.stamp.nanosec / 1000000000) 
        if self.latest_time_entered != -1: 
            delta_t = curr_time - self.latest_time_entered
            for parent, obj_desc in self.unified.obj_desc.get_parents(): 
                if obj_desc.get_time() + delta_t > self.params['sort.max_occlusion_time']: 
                    self.delete_association_tree(parent)
                else: 
                    obj_desc.set_time(obj_desc.get_time() + delta_t)
                    
                    # TODO write this function
                    # Collectively decrease confidence in object location at once for old detections
                    self.apply_kf(obj_desc)
        self.latest_time_entered = curr_time
                    
        for obj in global_desc_msg: 
            key = f'{obj.robot_id}.{obj.obj_id}'
            parent = self.unified.find(key)
            if parent == '': # Not in disjoint set
                self.unified.insert(obj, key)
            else: # In disjoint set somewhere 
                self.update_association_info(obj, parent, curr_time)
        
    def delete_association_tree(self, parent): 
        '''Delete association tree, because the set of detections has not been seen recently
        - parent: key of root node
        '''
        parent_info = self.unified.obj_desc[parent]
        
        for child in parent_info.children: 
            self.unified.delete(child)

        self.unified.delete(parent)
        
    def update_association_info(self, global_desc, parent_key, curr_time): 
        '''Assuming the key already exists in the set (i.e. was already added to the disjoint set):\n
        Update time, location
        '''
        self.unified.obj_desc[parent_key].time = curr_time
        self.unified.obj_desc[parent_key].dist = global_desc.distance
        self.unified.obj_desc[parent_key].pitch = global_desc.pitch
        self.unified.obj_desc[parent_key].yaw = global_desc.yaw
        
        if global_desc.max_confidence > self.unified.obj_desc[parent_key].descriptor_conf: 
            self.unified.obj_desc[parent_key].descriptor_conf = global_desc.max_confidence
            self.unified.obj_desc[parent_key].feature_desc = global_desc.best_descriptor

    def detect_inter_robot_associations(self): 
        '''Detect inter-robot object associations: runs every self.params['sort.re_cluster_secs'] 
        seconds from associations_ros_driver.py
        '''
        
        # neighbors_is_in_range: bool --> in range of any others?
        neighbors_is_in_range, neighbors_in_range_list = self.neighbor_manager.check_neighbors_in_range()
        
        if len(neighbors_in_range_list) > 0 and self.neighbor_manager.local_robot_is_broker():
            self.cluster(neighbors_in_range_list)

            # TODO get info every frame, and generalize StrongSORT + OSNet pipeline to relative poses from 
            # the perspective of the robot with the lowest agent ID (modifying Kalman Filter implementation) 
            # to deal with multi-agent associations

            # Apply some technique to send the resulting clusters out

    def cluster(self, curr_neighbors_in_range_list): 
        ''' Clusters all info of current neighbors in list
        - curr_neighbors_in_range_list: list of robot_id's, which is monotonically increasing
        '''
        latest_msg_time = -1
        abbreviated = [] # index is robot_id, value is list of keys with that robot_id
        for i in curr_neighbors_in_range_list: 
            abbreviated[i] = [key for key, value in self.unified.agents_seen.items() 
                              if len(value) < self.params['max_nb_robots'] and key.startswith(str(i))]

        for i in range(len(abbreviated) - 1): 
            first = abbreviated[i]
            second = abbreviated[i + 1]
                        
            for j in reversed(first): # j, k are string keys
                j_info = self.unified.obj_desc[j]
                
                if j_info.time > latest_msg_time: 
                    latest_msg_time = j_info.time

                if j_info.time < self.last_time_clustered: 
                    break
                
                heap = []
                for k in reversed(second): 
                    k_info = self.unified.obj_desc[k]

                    if k_info.time > latest_msg_time: 
                        latest_msg_time = k_info.time
                    
                    # Assuming each class is very different from each other class, don't cluster if detected as a 
                    # different class
                    if j_info.class_id != k_info.class_id: 
                        continue
                    
                    # r0_dist, r0_pitch, r0_yaw, r1_dist, r1_pitch, r1_yaw, colocalization
                    # TODO make new dictionary or something for colocalization
                    check_odom = self.check_same_location(j_info.dist, j_info.pitch, j_info.yaw, 
                                                          k_info.dist, k_info.pitch, k_info.yaw, 
                                                          j.colocalization)
                    
                    if not check_odom: 
                        continue
                    
                    # Assume there's only one bounding box around any given object, due to NMS working perfectly
                    check_feature_desc = self.check_features(j_info.feature_desc, k_info.feature_desc)
                    
                    if check_feature_desc > self.params['sort.compact_desc_min_similarity']: 
                        heapq.heappush(heap, (-1 * check_feature_desc, k_info.obj_id)) # heapq is a min heap, NOT a max heap
            
                if len(heap) != 0: 
                    _, closest_obj_id = heapq.heappop(heap)                
                    self.unified.union(j, f"{k.agent_perspective}.{closest_obj_id}")

        self.last_time_clustered = latest_msg_time

        
    def check_same_location(self, r0_dist, r0_pitch, r0_yaw, r1_dist, r1_pitch, r1_yaw, colocalization): 
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
        
        return distance.euclidean(r0_base, r1_trans_rot) < self.params['sort.location_epsilon']
        
        
    def check_features(self, r0_descriptor, r1_descriptor): 
        '''Get cosine similarity between the two descriptors
        '''
        return cosine_similarity(r0_descriptor, r1_descriptor)
    