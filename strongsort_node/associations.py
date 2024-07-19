#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped

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
        
        # Key: {robot_id (from)}; Value: {Key: {robot_id_to}; Value: {TransformStamped message}}
        # This is useless - TODO delete after integrating tf2 transforms
        self.current_transforms = {} 
        for i in range(self.params['max_nb_robots']): 
            self.current_transforms[i] = {}
        
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
                    
                    # TODO fix bug here 7/19/24
                    not_seen_clusters.remove(parent)
                    
                    # if len(self.unified.obj_desc[parent].children) > 0: 
                    #     num_parents_with_children += 1
                        
                self.update_colocalization(obj, new_time)
                
            # print(f"Num of parents with children: {num_parents_with_children}")
            
        for not_seen_keys in not_seen_clusters: 
            if new_time - self.unified.obj_desc[not_seen_keys].time > self.params['sort.max_occlusion_time']: 
                self.delete_association_tree(not_seen_keys)
                
            # else: 
            # TODO figure this out
            #     self.apply_kf(self.unified.obj_desc[not_seen_keys], dt)
            
        print(f"Unified: {self.unified.get_all_clustered_keys()}")

        
    # # Version 2
    # # Update last seen time of all info in unified set, and generalize old location with EKF
    # # Insert all new messages into unified disjoint set, if new; if old, update last seen time to curr
    # # For each new message: add to cluster
    # # If object has not been seen in [some amount of time], delete
    # def info_callback(self, global_desc_msg): 
    #     # print(f"Before insertion into unified: {self.unified.get_all_clustered_keys()}")

    #     curr_time = global_desc_msg.header.stamp.sec + (global_desc_msg.header.stamp.nanosec / 1000000000) 
    #     if self.last_time_entered != -1: # if this is not the first time the callback is called
    #         for parent in self.unified.get_parents_keys(): 
    #             robot_id = int(parent.split(".", 1)[0])
    #             # last time this specific parent or one of its children was entered into this set
    #             last_time_agent_entered_val = self.last_time_agent_entered[robot_id]
    #             delta_t = curr_time - last_time_agent_entered_val
    #             obj_desc = self.unified.obj_desc[parent]
                
    #             if delta_t > self.params['sort.max_occlusion_time']: 
    #                 print(f"Deleting {parent}")
    #                 self.delete_association_tree(parent)
    #             # else: 
    #             #     obj_desc.set_time(curr_time)
                    
    #                 # TODO deal with kalman filter later 
    #                 # self.apply_kf(obj_desc, delta_t)
                    
    #             self.last_time_agent_entered[robot_id] = curr_time
                
    #     self.last_time_entered = curr_time

    #     if len(global_desc_msg.descriptors) > 0: 
    #         # print(f"Size of descriptors: {len(global_desc_msg.descriptors)}")
    #         # print(f"Robot ID: {global_desc_msg.descriptors[0].robot_id}")
    #         self.last_time_agent_entered.update({global_desc_msg.descriptors[0].robot_id: curr_time})
            
    #         num_parents_with_children = 0
    #         for obj in global_desc_msg.descriptors: 
    #             key = f'{obj.robot_id}.{obj.obj_id}'
    #             parent = self.unified.find(key)
    #             if parent == '': # Not in disjoint set
    #                 self.unified.insert(obj, key, curr_time)
    #             else: # In disjoint set somewhere 
    #                 self.update_association_info(obj, parent, curr_time)
    #                 if len(self.unified.obj_desc[parent].children) > 0: 
    #                     num_parents_with_children += 1
                        
    #             self.update_colocalization(obj, curr_time)
                
    #         print(f"Num of parents with children: {num_parents_with_children}")
    #     print(f"Unified: {self.unified.get_all_clustered_keys()}")

    def delete_association_tree(self, parent): 
        '''Delete association tree, because the set of detections has not been seen recently
        - parent: key of root node
        '''
        parent_info = self.unified.obj_desc[parent]
        
        for child in parent_info.children: 
            self.unified.delete(child)

        self.unified.delete(parent)
        
    # TODO finish this function
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
            kf = obj.get_kalman_filter()

            kf.set_A(dt)
            
            # Might have to set this every single time
            if kf.get_update_num() == -1: 
                # Future work: replace velocity with twist from odometry message, and use rotation info
                if obj.robot_id == self.params['robot_id']: 
                    pose_now = self.pose_dict[f'{obj.robot_id}.{self.last_time_agent_entered[obj.robot_id]}']
                    pose_then = self.pose_dict[f'{obj.robot_id}.{obj.time}']

                    x_pos = pose_then.position.x - pose_now.position.x
                    y_pos = pose_then.position.y - pose_now.position.y
                    z_pos = pose_then.position.z - pose_now.position.z

                    kf.set_ref_frame(self.params['robot_id'], np.array([[x_pos], [y_pos], [z_pos], [0], [0], [0]]))
                else: 
                    # broker_pose_now = self.pose_dict[f'{self.params['robot_id']}.{self.last_time_agent_entered[obj.robot_id]}']
                    other_pose_now = self.pose_dict[f'{obj.robot_id}.{self.last_time_agent_entered[obj.robot_id]}']
                    other_pose_then = self.pose_dict[f'{obj.robot_id}.{obj.time}']

                    transform_stamped = self.current_transforms[self.params['robot_id']][obj.robot_id] 

                    x_pos = (transform_stamped.transform.translation.x + other_pose_then.position.x - other_pose_now.position.x)
                    y_pos = (transform_stamped.transform.translation.y + other_pose_then.position.y - other_pose_now.position.y)
                    z_pos = (transform_stamped.transform.translation.z + other_pose_then.position.z - other_pose_now.position.z)
                
                    kf.set_ref_frame(self.params['robot_id'], np.array([[x_pos], [y_pos], [z_pos], [0], [0], [0]]))
                
                kf.set_update_num(kf.get_update_num() + 1)
            # TODO figure out else block, and use this to change dist, pitch, yaw

            kf.predict_and_update(self.params['robot_id'])
                
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
            
    def update_colocalization(self, obj, curr_time): 
        self.pose_dict.update({f'{obj.robot_id}.{curr_time}': obj.pose})
        self.current_transforms[obj.robot_id].update({obj.robot_id_to: obj.colocalization})

    def detect_inter_robot_associations(self): 
        '''Detect inter-robot object associations: runs every self.params['sort.re_cluster_secs'] 
        seconds from associations_ros_driver.py
        '''
        
        # neighbors_is_in_range: bool --> in range of any others?
        neighbors_is_in_range, neighbors_in_range_list = self.neighbor_manager.check_neighbors_in_range()
        
        # neighbors_in_range_list will always have one element (same robot_id)
        if len(neighbors_in_range_list) > 1 and self.neighbor_manager.local_robot_is_broker():
            print(f"\nClustering from robot {self.params['robot_id']}: neighbors_is_in_range: {neighbors_is_in_range}\tneighbors_in_range_list: {neighbors_in_range_list}")
            self.cluster(set(neighbors_in_range_list))
            
            self.unify_clustered_objects(neighbors_in_range_list)

            print("\n")

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
        abbreviated = {}
        abbv_time = {}
        for i in range(self.params['max_nb_robots']): 
            abbreviated[i] = []
            abbv_time[i] = []
        
        for key in parent_keys_lst: 
            robot_id = int(key.split(".", 1)[0])
            # print(f"robot_id: {robot_id}\tcurr_neighbors_in_range_set: {curr_neighbors_in_range_set}")
            # print(f"robot_id in curr_neighbors_in_range_set: {robot_id in curr_neighbors_in_range_set}")
            # print(f"len(self.unified.obj_desc[key].children) < self.params['max_nb_robots'] - 1: {len(self.unified.obj_desc[key].children) < self.params['max_nb_robots'] - 1}")
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

                # One place where the ordered dict by insertion part comes from
                if j_info.time < self.last_time_clustered: 
                    break
                
                heap = []
                for k in reversed(second): 
                    k_info = self.unified.obj_desc[k]

                    if k_info.time > latest_msg_time: 
                        latest_msg_time = k_info.time
                        
                    # print(f"j: {j_info.robot_id}.{j_info.time}\tk: {k_info.robot_id}.{k_info.time}")
                    
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

        
    def check_same_location(self, r0_frame_id, r0_id, r0_time, r0_dist, r0_pitch, r0_yaw, 
                            r1_frame_id, r1_id, r1_time, r1_dist, r1_pitch, r1_yaw): 
        # Type: geometry_msgs/Transform
        colocalization = self.get_colocalize_transform(r0_frame_id, r0_id, r0_time, 
                                                       r1_frame_id, r1_id, r1_time)
        
        # Assumes user is looking forward
        # Using spherical coords: pitch is 0 (up) to 180 (down) degrees, yaw is 0 to 360 degrees
        r0_base = [r0_dist * math.cos(r0_pitch) * math.sin(r0_yaw), 
                   r0_dist * math.cos(r0_pitch) * math.cos(r0_yaw), 
                   r0_dist * math.cos(r0_pitch)]
        r1_base = [r1_dist * math.cos(r1_pitch) * math.sin(r1_yaw), 
                   r1_dist * math.cos(r1_pitch) * math.cos(r1_yaw), 
                   r1_dist * math.cos(r1_pitch)]

        quat_ros = colocalization.rotation
        quat = R.from_quat([quat_ros.x, quat_ros.y, quat_ros.z, quat_ros.w])
        
        print(f"Quat as mtx: {quat.as_matrix()}\tr0_base: {r0_base}")
        
        r1_rot = quat.as_matrix().dot(r1_base)
        
        r1_trans_rot = np.zeros(3)
        r1_trans_rot[0] = r1_rot[0] - colocalization.translation.x
        r1_trans_rot[1] = r1_rot[1] - colocalization.translation.y
        r1_trans_rot[2] = r1_rot[2] - colocalization.translation.z
        
        print(f'Transformed: {r1_trans_rot}')
        
        # TODO figure out why distance is > 2.5ish for each object
        dist = distance.euclidean(r0_base, r1_trans_rot)
        print(f"Distance between two objects: {dist}")
        
        return dist < self.params['sort.location_epsilon']
        
    # r0 will have a lower robot_id than r1                    
    def get_colocalize_transform(self, r0_frame_id, r0_id, r0_time, r1_frame_id, r1_id, r1_time): 
        '''Get transformation between the frames of two different agents at two different times, using 
        information on agent pose over time and current colocalization\n
        Eg: use dictionary for r0_(t-3) to r0_t, current colocalization to convert from r0_t to r1_t, 
        and dictionary from r1_t to r1_(t-5)
        '''
        # TODO test this
        try: 
            curr_transform = self.tf_buffer.lookup_transform(r0_frame_id, r1_frame_id, rclpy.time.Time())
            
            a_pose_then = self.pose_dict[f'{r0_id}.{r0_time}']
            b_pose_then = self.pose_dict[f'{r1_id}.{r1_time}']
            
            a_pose_now = self.pose_dict[f'{r0_id}.{self.last_time_agent_entered[r0_id]}']
            b_pose_now = self.pose_dict[f'{r1_id}.{self.last_time_agent_entered[r1_id]}']
            
            translation=Vector3(
                x=(a_pose_now.position.x - a_pose_then.position.x + 
                   curr_transform.transform.translation.x + b_pose_then.position.x - 
                   b_pose_now.position.x),
                y=(a_pose_now.position.y - a_pose_then.position.y + 
                   curr_transform.transform.translation.y + b_pose_then.position.y - 
                   b_pose_now.position.y),
                z=(a_pose_now.position.z - a_pose_then.position.z + 
                   curr_transform.transform.translation.z + b_pose_then.position.z - 
                   b_pose_now.position.z)
            )
            
            transform_quat_ros = curr_transform.transform.rotation
            transform_quat = R.from_quat([transform_quat_ros.x, transform_quat_ros.y, transform_quat_ros.z, transform_quat_ros.w])

            # TODO figure out quaternions here
            a_quat_ros_then = a_pose_then.orientation
            a_quat_then = R.from_quat([a_quat_ros_then.x, a_quat_ros_then.y, a_quat_ros_then.z, a_quat_ros_then.w])

            a_quat_ros_now = a_pose_now.orientation
            a_quat_now = R.from_quat([a_quat_ros_now.x, a_quat_ros_now.y, a_quat_ros_now.z, a_quat_ros_now.w])

            # Rot from a_then to a_now: then.inv() * now
            # Verification: then * then_to_now = now
            # Note: for all quaternions q, q = -q
            a_quat_then_to_now = a_quat_then.inv() * a_quat_now
            
            b_quat_ros_then = b_pose_then.orientation
            b_quat_then = R.from_quat([b_quat_ros_then.x, b_quat_ros_then.y, b_quat_ros_then.z, b_quat_ros_then.w])

            b_quat_ros_now = b_pose_now.orientation
            b_quat_now = R.from_quat([b_quat_ros_now.x, b_quat_ros_now.y, b_quat_ros_now.z, b_quat_ros_now.w])

            # Rot from b_now to b_then: now.inv() * then
            b_quat_now_to_then = b_quat_now.inv() * b_quat_then
            
            final_rot = a_quat_then_to_now * transform_quat * b_quat_now_to_then
            final_quat = final_rot.as_quat()
            final_quat_ros = Quaternion(x=final_quat[0], y=final_quat[1], z=final_quat[2], w=final_quat[3])
            
            return Transform(translation=translation, rotation=final_quat_ros)
            
        except Exception as e: 
            print(f"Exception with colocalization transformation: {e}")

        # # print(self.pose_dict.keys())
        # a_pose_then = self.pose_dict[f'{r0_id}.{r0_time}']
        # b_pose_then = self.pose_dict[f'{r1_id}.{r1_time}']
        
        # a_pose_now = self.pose_dict[f'{r0_id}.{self.last_time_agent_entered[r0_id]}']
        # b_pose_now = self.pose_dict[f'{r1_id}.{self.last_time_agent_entered[r1_id]}']
        
        # print(self.current_transforms)
        # transform_stamped = self.current_transforms[r0_id][r1_id] # from a_pose_now to b_pose_now
        
        # desired_transform = Transform(
        #     translation=Vector3(
        #         x=(a_pose_now.position.x - a_pose_then.position.x + 
        #            transform_stamped.transform.translation.x + b_pose_then.position.x - 
        #            b_pose_now.position.x),
        #         y=(a_pose_now.position.y - a_pose_then.position.y + 
        #            transform_stamped.transform.translation.y + b_pose_then.position.y - 
        #            b_pose_now.position.y),
        #         z=(a_pose_now.position.z - a_pose_then.position.z + 
        #            transform_stamped.transform.translation.z + b_pose_then.position.z - 
        #            b_pose_now.position.z),
        #     ), 
        #     rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        # )
        
        # return desired_transform
        
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