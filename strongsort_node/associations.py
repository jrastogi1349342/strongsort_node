#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
import math
import heapq

from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped
from scipy.spatial.transform import Rotation as R

from strongsort_msgs.msg import MOTGlobalDescriptors, UnifiedObjectIDs
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
        # Key: {robot_id}.{timestamp}
        # Value: geometry_msgs/Pose pose
        # TODO delete old messages to save memory
        self.pose_dict = {} 
        
        # Key: {robot_id (from)}; Value: {Key: {robot_id_to}; Value: {TransformStamped message}}
        self.current_transforms = {} 
        for i in range(self.params['max_nb_robots']): 
            self.current_transforms[i] = {}
        
        self.last_time_agent_entered = {}
        self.last_time_entered = -1 # last time new information entered via callback
        self.last_time_clustered = -1 # last time the information was clustered

        # Get info from all robots --> TODO fix this
        self.info_sub = self.node.create_subscription(
            MOTGlobalDescriptors, 
            f"/{self.params['name_space']}/mot", 
            self.info_callback, 
            100)
        
        # Goes to all robots but only the robot with the right robot ID uses it for anything
        self.unified_publishers_arr = []
        for id in range(self.params['max_nb_robots']): 
            self.unified_publishers_arr[id] = self.node.create_publisher(UnifiedObjectIDs, '/mot/unified', 10)

    # Version 2
    # Update last seen time of all info in unified set, and generalize old location with EKF
    # Insert all new messages into unified disjoint set, if new; if old, update last seen time to curr
    # For each new message: add to cluster
    # If object has not been seen in [some amount of time], delete
    def info_callback(self, global_desc_msg): 
        curr_time = global_desc_msg.header.stamp.sec + (global_desc_msg.header.stamp.nanosec / 1000000000) 
        if self.last_time_entered != -1: 
            delta_t = curr_time - self.last_time_entered
            for parent in self.unified.obj_desc.get_parents_keys(): 
                obj_desc = self.unified.obj_desc[parent]
                if obj_desc.get_time() + delta_t > self.params['sort.max_occlusion_time']: 
                    self.delete_association_tree(parent)
                else: 
                    obj_desc.set_time(curr_time)
                    self.apply_kf(obj_desc, delta_t)
        self.last_time_entered = curr_time

        self.last_time_agent_entered[global_desc_msg.descriptors[0].robot_id] = curr_time
        
        for obj in global_desc_msg.descriptors: 
            key = f'{obj.robot_id}.{obj.obj_id}'
            parent = self.unified.find(key)
            if parent == '': # Not in disjoint set
                self.unified.insert(obj, key, curr_time)
                self.update_colocalization(obj, curr_time)
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
        
    # TODO write this function
    # Collectively decrease confidence in object location at once for old detections
    def apply_kf(self, obj, dt):
        '''Applies Kalman Filter to all objects, to account for an increased amount of time between
        the last guaranteed information and now
        - obj: ObjectDescription object
        - dt: Change in time between last application of KF and now
        ''' 
        kf = obj.get_kalman_filter()

        kf.set_A(dt)
        
        if kf.get_update_num() == -1: 
            # TODO set x_vec and ref_frame
            
            # kf.set_ref_frame(self.)
            
            
            print("starting KF for first time for this object")

        # TODO continue this
        kf.predict()
        kf.update()
                
    def update_association_info(self, obj, parent_key, curr_time): 
        '''Assuming the key already exists in the set (i.e. was already added to the disjoint set):\n
        Update time, location
        '''
        self.unified.obj_desc[parent_key].time = curr_time
        self.unified.obj_desc[parent_key].dist = obj.distance
        self.unified.obj_desc[parent_key].pitch = obj.pitch
        self.unified.obj_desc[parent_key].yaw = obj.yaw
        
        self.update_colocalization(obj, curr_time)
        
        if obj.max_confidence > self.unified.obj_desc[parent_key].descriptor_conf: 
            self.unified.obj_desc[parent_key].descriptor_conf = obj.max_confidence
            self.unified.obj_desc[parent_key].feature_desc = obj.best_descriptor
            
    def update_colocalization(self, obj, curr_time): 
        self.pose_dict.update({f'{obj.robot_id}.{curr_time}': obj.pose})
        self.current_transforms[obj.robot_id].update({f'{obj.robot_id_to}': obj.colocalization})

    def detect_inter_robot_associations(self): 
        '''Detect inter-robot object associations: runs every self.params['sort.re_cluster_secs'] 
        seconds from associations_ros_driver.py
        '''
        
        # neighbors_is_in_range: bool --> in range of any others?
        neighbors_is_in_range, neighbors_in_range_list = self.neighbor_manager.check_neighbors_in_range()
        
        # neighbors_in_range_list will always have one element (same robot_id)
        if len(neighbors_in_range_list) > 1 and self.neighbor_manager.local_robot_is_broker():
            self.cluster(set(neighbors_in_range_list))
            
            self.unify_clustered_objects(neighbors_in_range_list)
            
            self.publish_unified()

            # TODO get info every frame, and generalize StrongSORT + OSNet pipeline to relative poses from 
            # the perspective of the robot with the lowest agent ID (modifying Kalman Filter implementation) 
            # to deal with multi-agent associations

    def cluster(self, curr_neighbors_in_range_set): 
        ''' Clusters all info of current neighbors in list
        - curr_neighbors_in_range_set: set of robot_id's
        '''
        latest_msg_time = -1
        parent_keys_lst = self.unified.get_parents_keys()
        
        # index is robot_id, value is list of keys with that robot_id as the parent that have not yet been 
        # unified between all agents, ordered by insertion order
        abbreviated = [] 
        for key in parent_keys_lst: 
            robot_id = key.split(".", 1)[0]
            if (robot_id in curr_neighbors_in_range_set and 
                len(self.unified.obj_desc[key].children) < self.params['max_nb_robots'] - 1): 
                abbreviated[i].append(key)
                    
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
                    
                    # r0_id, r0_time, r0_dist, r0_pitch, r0_yaw, 
                    # r1_id, r1_time, r1_dist, r1_pitch, r1_yaw
                    check_odom = self.check_same_location(j_info.robot_id, j_info.time, 
                                                          j_info.dist, j_info.pitch, j_info.yaw, 
                                                          k_info.robot_id, k_info.time, 
                                                          k_info.dist, k_info.pitch, k_info.yaw)
                    
                    if not check_odom: 
                        continue
                    
                    # Assume there's only one bounding box around any given object, due to NMS working perfectly
                    check_feature_desc = self.check_features(j_info.feature_desc, k_info.feature_desc)
                    
                    if check_feature_desc > self.params['sort.compact_desc_min_similarity']: 
                        heapq.heappush(heap, (-1 * check_feature_desc, k_info.obj_id)) # heapq is a min heap, NOT a max heap
            
                if len(heap) != 0: 
                    _, closest_obj_id = heapq.heappop(heap)                
                    self.unified.union(j, f"{k.robot_id}.{closest_obj_id}")

        self.last_time_clustered = latest_msg_time

        
    def check_same_location(self, r0_id, r0_time, r0_dist, r0_pitch, r0_yaw, 
                            r1_id, r1_time, r1_dist, r1_pitch, r1_yaw): 
        colocalization = self.get_colocalize_transform(r0_id, r0_time, r1_id, r1_time)
        
        # Assumes user is looking forward
        # Using spherical coords: pitch is 0 (up) to 180 (down) degrees, yaw is 0 to 360 degrees
        r0_base = [r0_dist * math.cos(r0_pitch) * math.sin(r0_yaw), 
                   r0_dist * math.cos(r0_pitch) * math.cos(r0_yaw), 
                   r0_dist * math.cos(r0_pitch)]
        r1_base = [r1_dist * math.cos(r1_pitch) * math.sin(r1_yaw), 
                   r1_dist * math.cos(r1_pitch) * math.cos(r1_yaw), 
                   r1_dist * math.cos(r1_pitch)]
        
        r1_trans = []
        r1_trans[0] = r1_base[0] - colocalization.transform.translation.x
        r1_trans[1] = r1_base[1] - colocalization.transform.translation.y
        r1_trans[2] = r1_base[2] - colocalization.transform.translation.z

        quat_ros = colocalization.transform.rotation
        quat = R.from_quat([quat_ros.x, quat_ros.y, quat_ros.z, quat_ros.w])
        
        r1_trans_rot = quat.as_matrix * r1_trans
        
        return distance.euclidean(r0_base, r1_trans_rot) < self.params['sort.location_epsilon']
        
    # r0 will have a lower robot_id than r1                    
    def get_colocalize_transform(self, r0_id, r0_time, r1_id, r1_time): 
        '''Get transformation between the frames of two different agents at two different times, using 
        information on agent pose over time and current colocalization\n
        Eg: use dictionary for r0_(t-3) to r0_t, current colocalization to convert from r0_t to r1_t, 
        and dictionary from r1_t to r1_(t-5)
        '''
        a_pose_then = self.pose_dict[f'{r0_id}.{r0_time}']
        b_pose_then = self.pose_dict[f'{r1_id}.{r1_time}']
        
        a_pose_now = self.pose_dict[f'{r0_id}.{self.last_time_entered[r0_id]}']
        b_pose_now = self.pose_dict[f'{r1_id}.{self.last_time_entered[r1_id]}']
                
        transform_stamped = self.current_transforms[r0_id][r1_id] # from a_pose_now to b_pose_now
        
        # TODO figure out quaternion here
        desired_transform = Transform(
            translation=Vector3(
                x=(a_pose_now.position.x - a_pose_then.position.x + 
                   transform_stamped.transform.translation.x + b_pose_then.position.x - 
                   b_pose_now.position.x),
                y=(a_pose_now.position.y - a_pose_then.position.y + 
                   transform_stamped.transform.translation.y + b_pose_then.position.y - 
                   b_pose_now.position.y),
                z=(a_pose_now.position.z - a_pose_then.position.z + 
                   transform_stamped.transform.translation.z + b_pose_then.position.z - 
                   b_pose_now.position.z),
            ), 
            orientation=Quaternion()
        )
        
        return desired_transform
        
    def check_features(self, r0_descriptor, r1_descriptor): 
        '''Get cosine similarity between the two descriptors
        '''
        return cosine_similarity(r0_descriptor, r1_descriptor)
    
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
            all_keys_in_cluster = self.unified.get_all_clustered_keys(key)
            id_from_broker = self.unified.get_obj_id_in_cluster(key, self.params['robot_id'])
            
            if id_from_broker == -1: 
                for cluster_key in all_keys_in_cluster: 
                    info_arr = cluster_key.split(".", 1)
                    if info_arr[0] in neighbors_in_range_list: 
                        unified_mapping[info_arr[0]].update({info_arr[1]: unified_id_no_broker})
                        
                unified_id_no_broker -= 1
            else: 
                for cluster_key in all_keys_in_cluster: 
                    info_arr = cluster_key.split(".", 1)
                    if info_arr[0] in neighbors_in_range_list: 
                        unified_mapping[info_arr[0]].update({info_arr[1]: id_from_broker})
        
        for robot_id, mapping in unified_mapping.items(): 
            obj_ids = list(mapping.keys())            
            unified_obj_ids = list(mapping.values())
            
            self.unified_publishers_arr[robot_id].publish(UnifiedObjectIDs(
                robot_id=robot_id, 
                obj_ids=obj_ids, 
                unified_obj_ids=unified_obj_ids
            ))