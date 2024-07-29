#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.time import Time, Duration
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_vector3
from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped, Pose

from strongsort_msgs.msg import MOTGlobalDescriptors, UnifiedObjectIDs
from strongsort_node.neighbors_manager import NeighborManager
from strongsort_node.disjoint_set_associations import DisjointSetAssociations

import math
import heapq
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

class ObjectAssociation(object): 
    def __init__(self, params, node):
        self.params = params
        self.node = node
        self.neighbor_manager = NeighborManager(self.node, self.params)

        self.unified = DisjointSetAssociations()
        
        # Key: {robot_id}.{timestamp}
        # Value: geometry_msgs/Pose pose
        # TODO delete old messages to save memory
        self.pose_dict = {} 

        # Key: {robot_id}
        # Value: string frame id
        self.frame_ids = {}
                
        self.last_time_agent_entered = {}
        self.last_time_entered = -1 # last time new information entered via callback
        self.last_time_clustered = -1 # last time the information was clustered

        # Get info from all robots
        self.info_sub = self.node.create_subscription(
            MOTGlobalDescriptors, 
            f"/mot/descriptors", 
            self.info_callback, 
            100)
        
        # Get info about transforms between agents
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        
        # Goes to all robots but only the robot with the right robot ID uses it for anything
        self.unified_publishers_arr = {}
        for id in range(self.params['max_nb_robots']): 
            self.unified_publishers_arr[id] = self.node.create_publisher(UnifiedObjectIDs, '/mot/unified', 10)

    # Version 3
    def info_callback(self, descriptor_msg): 
        '''Callback for getting MOTGlobalDescriptors object from some robot
        - descriptor_msg: MOTGlobalDescriptors object; is from 1 unique robot ID 
        '''
        not_seen_clusters = set(self.unified.get_parents_keys())
        new_time = descriptor_msg.header.stamp.sec + (descriptor_msg.header.stamp.nanosec / 1000000000) 
        if len(descriptor_msg.descriptors) > 0: 
            # Update latest time broker got a message from this specific agent
            self.last_time_agent_entered.update({descriptor_msg.descriptors[0].robot_id: new_time})
            
            # num_parents_with_children = 0
            for obj in descriptor_msg.descriptors: 
                key = f'{obj.robot_id}.{obj.obj_id}'
                parent = self.unified.find(key)
                if parent == '': # Not in disjoint set
                    self.unified.insert(obj, key, new_time)
                else: # In disjoint set somewhere 
                    self.update_association_info(obj, parent, new_time)
                    
                    # Perhaps fixed bug, figure out when using kalman filter
                    if parent in not_seen_clusters: 
                        not_seen_clusters.remove(parent)
                                            
                self.pose_dict.update({f'{obj.robot_id}.{new_time}': obj.pose})
                
                if obj.robot_id not in self.frame_ids: 
                    self.frame_ids.update({obj.robot_id: obj.header.frame_id})
                            
        print(f"Not seen clusters: {not_seen_clusters}")
        for not_seen_keys in not_seen_clusters: 
            dt = new_time - self.unified.obj_desc[not_seen_keys].time
            if dt > self.params['sort.max_occlusion_time']: 
                self.delete_association_tree(not_seen_keys)
                
            # TODO test this and debug
            else: 
                self.apply_kf(self.unified.obj_desc[not_seen_keys], dt)
            
        print(f"Unified: {self.unified.get_all_clustered_keys()}")

    def delete_association_tree(self, parent): 
        '''Delete association tree, because the set of detections has not been seen recently
        - parent: key of root node
        '''
        parent_info = self.unified.obj_desc[parent]
        
        for child in parent_info.children: 
            self.unified.delete(child)

        self.unified.delete(parent)
        
    # TODO debug
    # Collectively decrease confidence in object location at once for old detections/use the fact that confidence isn't 
    # the highest to update kf in some other way
    def apply_kf(self, obj, dt):
        '''Applies Kalman Filter to all objects, to account for an increased amount of time between
        the last guaranteed information and now\n
        NOTE: this implementation is unstable in cases where the lowest ID in range changes dynamically, and 
        may have some multithreading bugs between this and the clustering algorithm
        - obj: ObjectDescription object
        - dt: Change in time between last application of KF and now
        ''' 
        _, neighbors_in_range_list = self.neighbor_manager.check_neighbors_in_range()

        if len(neighbors_in_range_list) > 1 and self.neighbor_manager.local_robot_is_broker():
            kf = obj.kalman_filter

            # Future work: include rotation info
            if kf.broker_id == self.params['robot_id']: 
                now_pose = self.pose_dict[f'{obj.robot_id}.{self.last_time_agent_entered[obj.robot_id]}']
                
                loc = Vector3(x=obj.dist * math.sin(obj.pitch) * math.cos(obj.yaw), 
                        y=obj.dist * math.sin(obj.pitch) * math.sin(obj.yaw), 
                        z=obj.dist * math.cos(obj.pitch))
                
                loc_trans = do_transform_vector3(loc, now_pose)
                
                # Object pose
                x = loc_trans.x + now_pose.position.x
                y = loc_trans.y + now_pose.position.y
                z = loc_trans.z + now_pose.position.z
                
                measurement = np.array([x, y, z, 0.0, 0.0, 0.0])
                kf.predict(dt)
                kf.update(measurement)
                                
            else: 
                broker_now_pose = self.pose_dict[f'{self.params['robot_id']}.{self.last_time_agent_entered[self.params['robot_id']]}']
                broker_now_time = self.time_float_to_time(self.last_time_agent_entered[self.params['robot_id']])
                
                # other_pose_then = self.pose_dict[f'{obj.robot_id}.{obj.time}']
                other_now_time = self.time_float_to_time(obj.time)

                # Create array mapping each agent ID to its frame_id, and access here as source frame
                broker_now_to_other_frame_then = self.tf_buffer.lookup_transform_full(
                    obj.frame_id, other_now_time, 
                    self.frame_ids[self.params['robot_id']], broker_now_time, 
                    "world", Duration(seconds=0.05)).transform
                
                other_then_pose = self.apply_pose_transformation(broker_now_to_other_frame_then, broker_now_pose)

                loc = Vector3(x=obj.dist * math.sin(obj.pitch) * math.cos(obj.yaw), 
                        y=obj.dist * math.sin(obj.pitch) * math.sin(obj.yaw), 
                        z=obj.dist * math.cos(obj.pitch))
                
                loc_trans = do_transform_vector3(loc, other_then_pose)
                
                # Object pose
                x = loc_trans.x + other_then_pose.position.x
                y = loc_trans.y + other_then_pose.position.y
                z = loc_trans.z + other_then_pose.position.z

                measurement = np.array([x, y, z, 0.0, 0.0, 0.0])
                kf.predict(dt)
                kf.update(measurement)
                
    def time_float_to_time(self, val): 
        sec = math.floor(val)
        ns = val - sec
        return Time(seconds=sec, nanoseconds=ns)
                
    def update_association_info(self, obj, parent_key, curr_time): 
        '''Assuming the key already exists in the set (i.e. was already added to the disjoint set):\n
        Update time, location
        '''
        self.unified.obj_desc[parent_key].time = curr_time
        self.unified.obj_desc[parent_key].dist = obj.distance
        self.unified.obj_desc[parent_key].pitch = obj.pitch
        self.unified.obj_desc[parent_key].yaw = obj.yaw
                
        if obj.max_confidence > self.unified.obj_desc[parent_key].descriptor_conf: 
            self.unified.obj_desc[parent_key].descriptor_conf = obj.max_confidence
            self.unified.obj_desc[parent_key].feature_desc = list(obj.best_descriptor)
            
    def detect_inter_robot_associations(self): 
        '''Detect inter-robot object associations: runs every self.params['sort.re_cluster_secs'] 
        seconds from associations_ros_driver.py
        '''
        
        self.single_agent_tests()
        
        # neighbors_is_in_range: bool --> in range of any others?
        neighbors_is_in_range, neighbors_in_range_list = self.neighbor_manager.check_neighbors_in_range()
        
        # neighbors_in_range_list will always have one element (same robot_id)
        if len(neighbors_in_range_list) > 1 and self.neighbor_manager.local_robot_is_broker():
            print(f"\nClustering from robot {self.params['robot_id']}: neighbors_is_in_range: {neighbors_is_in_range}\tneighbors_in_range_list: {neighbors_in_range_list}")
            self.cluster(set(neighbors_in_range_list))
            
            self.unify_clustered_objects(neighbors_in_range_list)

            print("\n")

    def single_agent_tests(self): 
        first_key = self.unified.get_parents_keys()[0]
        first_key_info = self.unified.obj_desc[first_key]
        
        self.single_agent_location_test(self, first_key_info.time, first_key_info.frame_id, 
                                        first_key_info.dist, first_key_info.pitch, first_key_info.yaw)

    def cluster(self, curr_neighbors_in_range_set): 
        ''' Clusters all info of current neighbors in list
        - curr_neighbors_in_range_set: set of robot_id's
        '''
        latest_msg_time = -1
        parent_keys_lst = self.unified.get_parents_keys()
        
        # index is robot_id, value is list of keys with that robot_id as the parent that have not yet been 
        # unified between all agents, ordered by insertion order
        abbreviated = {}
        abbv_time = {}
        for i in range(self.params['max_nb_robots']): 
            abbreviated[i] = []
            abbv_time[i] = []
        
        for key in parent_keys_lst: 
            robot_id = int(key.split(".", 1)[0])

            if (robot_id in curr_neighbors_in_range_set and 
                len(self.unified.obj_desc[key].children) < self.params['max_nb_robots'] - 1): 
                abbreviated[robot_id].append(key)
                abbv_time[robot_id].append((key, self.unified.obj_desc[key].time))
                
        print(f"Abbreviated items to cluster: {abbreviated}")
        print(f"Abbreviated + time: {abbv_time}")
                    
        for i in range(len(abbreviated) - 1): 
            first = abbreviated[i] # robot 0
            second = abbreviated[i + 1] # robot 1
                        
            for j in reversed(first): # j, k are string keys
                j_info = self.unified.obj_desc[j]
                
                if j_info.time > latest_msg_time: 
                    latest_msg_time = j_info.time

                # One place where the ordered dict by insertion part is necessary
                if j_info.time < self.last_time_clustered: 
                    break
                
                heap = []
                for k in reversed(second): 
                    k_info = self.unified.obj_desc[k]

                    if k_info.time > latest_msg_time: 
                        latest_msg_time = k_info.time
                                            
                    # Assuming each class is very different from each other class, don't cluster if detected as a 
                    # different class
                    if j_info.robot_id == k_info.robot_id or j_info.class_id != k_info.class_id: 
                        continue
                    
                    # r0_id, r0_time, r0_dist, r0_pitch, r0_yaw, 
                    # r1_id, r1_time, r1_dist, r1_pitch, r1_yaw
                    check_odom = self.check_same_location(j_info.frame_id, j_info.robot_id, j_info.time, 
                                                          j_info.dist, j_info.pitch, j_info.yaw, 
                                                          k_info.frame_id, k_info.robot_id, k_info.time, 
                                                          k_info.dist, k_info.pitch, k_info.yaw)
                    
                    # Temp removed to test feature description + union - TODO re-add
                    print(f"Check odom: {check_odom}")
                    # if not check_odom: 
                    #     continue
                    
                    # Assume there's only one bounding box around any given object, due to NMS working perfectly
                    check_feature_desc = self.check_features(j_info.feature_desc, k_info.feature_desc)
                    
                    if check_feature_desc > self.params['sort.compact_desc_min_similarity']: 
                        heapq.heappush(heap, (-1 * check_feature_desc, k_info.obj_id)) # heapq is a min heap, NOT a max heap
            
                if len(heap) != 0: 
                    _, closest_obj_id = heapq.heappop(heap)
                    self.unified.union(j, f"{i + 1}.{closest_obj_id}")

        self.last_time_clustered = latest_msg_time
        
    def single_agent_location_test(self, r0_time, r0_frame_id, r0_dist, r0_pitch, r0_yaw): 
        try: 
            A_now_pose = self.pose_dict[f'0.{self.last_time_agent_entered[0]}']
            A_now_time = self.last_time_agent_entered[0]
            
            r0_sec = math.floor(r0_time)
            r0_ns = r0_time - r0_sec
            r0_when = Time(seconds=r0_sec, nanoseconds=r0_ns)
            
            A_now_to_r0_frame_then = self.tf_buffer.lookup_transform_full(
                r0_frame_id, r0_when, 
                "A_odom", A_now_time, 
                "world", Duration(seconds=0.05)).transform
            
            r0_then_pose = self.apply_pose_transformation(A_now_to_r0_frame_then, A_now_pose)

            r0_loc = Vector3(x=r0_dist * math.sin(r0_pitch) * math.cos(r0_yaw), 
                      y=r0_dist * math.sin(r0_pitch) * math.sin(r0_yaw), 
                      z=r0_dist * math.cos(r0_pitch))
            
            r0_loc_trans = do_transform_vector3(r0_loc, r0_then_pose)
            
            r0_x = r0_then_pose.position.x + r0_loc_trans.x
            r0_y = r0_then_pose.position.y + r0_loc_trans.y
            r0_z = r0_then_pose.position.z + r0_loc_trans.z
            
            r0_obj_loc = np.array([r0_x, r0_y, r0_z])
            
            print(r0_obj_loc)
                        
        except Exception as e: 
            print(f"Exception with transformation: {e}")


    def check_same_location(self, r0_frame_id, r0_id, r0_time, r0_dist, r0_pitch, r0_yaw, r0_x, r0_y, r0_z, 
                            r1_frame_id, r1_id, r1_time, r1_dist, r1_pitch, r1_yaw, r1_x, r1_y, r1_z): 
        '''Check if location of object is same between two robots and two timestamps\n
        Convention is that translations are applied before rotations in Transform messages\n
        NOTE: won't scale if A is not present in range
        '''
        try: 
            broker_now_pose = self.pose_dict[f'{self.params['robot_id']}.{self.last_time_agent_entered[0]}']
            broker_now_time = self.last_time_agent_entered[self.params['robot_id']]
            
            r0_sec = math.floor(r0_time)
            r0_ns = r0_time - r0_sec
            r0_when = Time(seconds=r0_sec, nanoseconds=r0_ns)
            
            broker_now_to_r0_frame_then = self.tf_buffer.lookup_transform_full(
                r0_frame_id, r0_when, 
                self.frame_ids[self.params['robot_id']], broker_now_time, 
                "world", Duration(seconds=0.05)).transform
            
            r0_then_pose = self.apply_pose_transformation(broker_now_to_r0_frame_then, broker_now_pose)

            r0_loc = Vector3(x=r0_x, y=r0_y, z=r0_z)
            # r0_loc = Vector3(x=r0_dist * math.sin(r0_pitch) * math.cos(r0_yaw), 
            #           y=r0_dist * math.sin(r0_pitch) * math.sin(r0_yaw), 
            #           z=r0_dist * math.cos(r0_pitch))
            
            r0_loc_trans = do_transform_vector3(r0_loc, r0_then_pose)
            
            # Object pose
            r0_x = r0_then_pose.position.x + r0_loc_trans.x
            r0_y = r0_then_pose.position.y + r0_loc_trans.y
            r0_z = r0_then_pose.position.z + r0_loc_trans.z
            
            r0_obj_loc = np.array([r0_x, r0_y, r0_z])
            
            # --------------r1-----------------

            
            r1_sec = math.floor(r1_time)
            r1_ns = r1_time - r1_sec
            r1_when = Time(seconds=r1_sec, nanoseconds=r1_ns)
            
            broker_now_to_r1_frame_then = self.tf_buffer.lookup_transform_full(
                r1_frame_id, r1_when, 
                self.frame_ids[self.params['robot_id']], broker_now_time, 
                "world", Duration(seconds=0.05)).transform
            
            r1_then_pose = self.apply_pose_transformation(broker_now_to_r1_frame_then, broker_now_pose)

            r1_loc = Vector3(x=r1_x, y=r1_y, z=r1_z)
            # r1_loc = Vector3(x=r1_dist * math.sin(r1_pitch) * math.cos(r1_yaw), 
            #           y=r1_dist * math.sin(r1_pitch) * math.sin(r1_yaw), 
            #           z=r1_dist * math.cos(r1_pitch))
            
            r1_loc_trans = do_transform_vector3(r1_loc, r1_then_pose)
            
            # Object pose
            r1_x = r1_then_pose.position.x + r1_loc_trans.x
            r1_y = r1_then_pose.position.y + r1_loc_trans.y
            r1_z = r1_then_pose.position.z + r1_loc_trans.z
            
            r1_obj_loc = np.array([r1_x, r1_y, r1_z])

            
            dist = distance.euclidean(r0_obj_loc, r1_obj_loc)
            print(f"Distance between obj from r0 ({r0_obj_loc}) and 
                  r1 ({r1_obj_loc}): {dist}")
            
            return dist < self.params['sort.location_epsilon']


        except Exception as e: 
            print(f"Exception with transformation: {e}")
    
    def apply_pose_transformation(self, transform, pose): 
        '''Apply transform A -> B to pose A to get pose B
        - transform: geometry_msgs/Transform message of A -> B
        - pose: geometry_msgs/Pose message of A
        '''
        new_pose = Pose()
        
        new_pose.position.x = pose.position.x + transform.translation.x
        new_pose.position.y = pose.position.y + transform.translation.y
        new_pose.position.z = pose.position.z + transform.translation.z
        
        trans_quat_ros = transform.rotation
        trans_quat = R.from_quat([trans_quat_ros.x, trans_quat_ros.y, trans_quat_ros.z, trans_quat_ros.w])

        pose_quat_ros = pose.orientation
        pose_quat = R.from_quat([pose_quat_ros.x, pose_quat_ros.y, pose_quat_ros.z, pose_quat_ros.w])
        
        new_quat = trans_quat * pose_quat
        new_pose.orientation = Quaternion(x=new_quat.x, y=new_quat.y, z=new_quat.z, w=new_quat.w)
        
        return new_pose
                
    def check_features(self, r0_descriptor, r1_descriptor): 
        '''Get cosine similarity between the two descriptors
        '''
        sim = np.dot(r0_descriptor, r1_descriptor)/(norm(r0_descriptor) * norm(r1_descriptor))
        print(f'Similarity between r0 and r1: {sim}')
        
        return sim
    
    def unify_clustered_objects(self, neighbors_in_range_list): 
        '''Creates unified labeling for all clusters and publishes it: \n
        If broker has already seen the value, set that as the unified object ID; 
        otherwise set it to a negative value that gets incremented constantly
        '''        
        unified_mapping = {} # Key: {robot_id}; Value: {Key: {obj_id}; Value: {unified_obj_id}}
        for id in neighbors_in_range_list: 
            unified_mapping[id] = {}
            
        # This will construct the unified IDs where the broker hasn't seen the object before
        # Decrement by 1 for each new object
        unified_id_no_broker = -1
        
        parents = self.unified.get_parents_keys()
        for key in parents: 
            all_keys_in_cluster = self.unified.get_keys_in_cluster(key)
            obj_id_from_broker = self.unified.get_obj_id_in_cluster(key, self.params['robot_id'])
            
            if obj_id_from_broker == -1: 
                for cluster_key in all_keys_in_cluster: 
                    info_arr = cluster_key.split(".", 1)
                    if info_arr[0] in neighbors_in_range_list: 
                        unified_mapping[info_arr[0]].update({info_arr[1]: unified_id_no_broker})
                        
                unified_id_no_broker -= 1
            else: 
                for cluster_key in all_keys_in_cluster: 
                    info_arr = cluster_key.split(".", 1)
                    if info_arr[0] in neighbors_in_range_list: 
                        unified_mapping[info_arr[0]].update({info_arr[1]: obj_id_from_broker})
        
        for robot_id, mapping in unified_mapping.items(): 
            obj_ids = list(mapping.keys())            
            unified_obj_ids = list(mapping.values())
            
            self.unified_publishers_arr[robot_id].publish(UnifiedObjectIDs(
                robot_id=robot_id, 
                obj_ids=obj_ids, 
                unified_obj_ids=unified_obj_ids
            ))