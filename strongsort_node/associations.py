#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.time import Time, Duration
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_vector3
from geometry_msgs.msg import Vector3, Vector3Stamped, Quaternion, Transform, TransformStamped, Pose

from strongsort_msgs.msg import MOTGlobalDescriptors, UnifiedObjectIDs
from strongsort_node.neighbors_manager import NeighborManager
from strongsort_node.disjoint_set_associations import DisjointSetAssociations
from strongsort_node.conf_kalman_filter import gaussian_bhattacharyya

import math
import heapq
import numpy as np
from numpy.linalg import norm
# from scipy.spatial.transform import Rotation as R
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import traceback

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

    # TODO figure out how to deal with YOLO jitters (split second detections as a different class, 
    # leading to having multiple clusters for the same object)
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
                    dt = new_time - self.unified.obj_desc[parent].time
                    self.update_association_info(obj, parent, new_time, dt)
                    
                    # Perhaps fixed bug, figure out when using kalman filter
                    if parent in not_seen_clusters: 
                        not_seen_clusters.remove(parent)
                                            
                self.pose_dict.update({f'{obj.robot_id}.{new_time}': obj.pose})
                
                if obj.robot_id not in self.frame_ids: 
                    self.frame_ids.update({obj.robot_id: obj.header.frame_id})
                            
        # print(f"Not seen clusters: {not_seen_clusters}")
        for not_seen_keys in not_seen_clusters: 
            dt = new_time - self.unified.obj_desc[not_seen_keys].time
            if dt > self.params['sort.max_occlusion_time']: 
                self.delete_association_tree(not_seen_keys)
                
            else: 
                # TODO revert comments and debug
                kf = self.unified.obj_desc[not_seen_keys].kalman_filter
                kf_dt = new_time - kf.last_updated_time
                # kf.predict(kf_dt)
                kf.last_updated_time = new_time
            
        print(f"Unified: {self.unified.get_all_clustered_keys()}")

    def delete_association_tree(self, parent): 
        '''Delete association tree, because the set of detections has not been seen recently
        - parent: key of root node
        '''
        parent_info = self.unified.obj_desc[parent]
        
        for child in parent_info.children: 
            self.unified.delete(child)

        self.unified.delete(parent)
        
    def update_association_info(self, obj, parent_key, curr_time, dt): 
        '''Assuming the key already exists in the set (i.e. was already added to the disjoint set):\n
        Update time, location
        '''
        obj_desc = self.unified.obj_desc[parent_key]
        obj_desc.time = curr_time
                
        if obj.max_confidence > obj_desc.descriptor_conf: 
            obj_desc.descriptor_conf = obj.max_confidence
            obj_desc.feature_desc = list(obj.best_descriptor)
            
        # print(f"Updated position: \tx: {obj.rel_x}\ty: {obj.rel_y}\tz: {obj.rel_z}")
            
        obj_desc.kalman_filter.last_updated_time = curr_time
            
        obj_desc.kalman_filter.x[0] = obj.rel_x
        obj_desc.kalman_filter.x[1] = obj.rel_y
        obj_desc.kalman_filter.x[2] = obj.rel_z
                    
        # TODO remove above x, y, z assignment and debug this
        # self.full_kf(obj, parent_key, dt, obj.rel_x, obj.rel_y, obj.rel_z)
                
    # TODO debug, and update to how check_same_location and single_agent_location_test are written
    def full_kf(self, obj, parent_key, dt, new_x, new_y, new_z):
        '''Applies Kalman Filter to all objects, to account for an increased amount of time between
        the last guaranteed information and now\n
        NOTE: this implementation is unstable in cases where the lowest ID in range changes dynamically, and 
        may have some multithreading bugs between this and the clustering algorithm
        - obj: ObjectDescription object
        - parent_key: Key of parent node in this cluster
        - dt: Change in time between last application of KF and now
        - new_x: New x position of object, to update Kalman Filter
        - new_y: New y position of object, to update Kalman Filter
        - new_z: New z position of object, to update Kalman Filter
        ''' 
        _, neighbors_in_range_list = self.neighbor_manager.check_neighbors_in_range()
        
        kf = self.unified.obj_desc[parent_key].kalman_filter

        if self.neighbor_manager.local_robot_is_broker(): 
            x = kf.x[0]
            y = kf.x[1]
            z = kf.x[2]
            
            print(f"Key: {parent_key}\tBroker id: {kf.broker_id}")

            # Future work: include rotation info
            # In this case, there are >= 1 robots in range
            if kf.broker_id == self.params['robot_id']: 
                x = new_x
                y = new_y
                z = new_z
                # now_pose = self.pose_dict[f'{obj.robot_id}.{self.last_time_agent_entered[obj.robot_id]}']
                
                # loc = Vector3(x=new_x, y=new_y, z=new_z)
                
                # loc_trans = do_transform_vector3(loc, now_pose)
                
                # # Object pose
                # x = loc_trans.x + now_pose.position.x
                # y = loc_trans.y + now_pose.position.y
                # z = loc_trans.z + now_pose.position.z
                
            # If not this robot, then there are > 1 robots in range
            else: 
                print(f"\nself.last_time_agent_entered: {self.last_time_agent_entered}\n")
                id = self.params['robot_id']
                last_time_entered = self.last_time_agent_entered[id]
                parent_now_time = self.time_float_to_time(last_time_entered)
                
                other_now_time = self.time_float_to_time(
                    obj.header.stamp.sec + (obj.header.stamp.nanosec / 1000000000))

                # Create array mapping each agent ID to its frame_id, and access here as source frame
                parent_now_to_other_frame_then = self.tf_buffer.lookup_transform_full(
                    self.frame_ids[id], parent_now_time, 
                    obj.header.frame_id, other_now_time, 
                    "world", Duration(seconds=0.5))
                
                loc = Vector3Stamped(vector=Vector3(x=new_x, y=new_y, z=new_z))
                
                loc_trans = do_transform_vector3(loc, parent_now_to_other_frame_then)
                
                # Object pose
                x = loc_trans.vector.x + parent_now_to_other_frame_then.transform.translation.x
                y = loc_trans.vector.y + parent_now_to_other_frame_then.transform.translation.y
                z = loc_trans.vector.z + parent_now_to_other_frame_then.transform.translation.z

            measurement = np.array([x, y, z])
            kf.predict(dt)
            kf.update(measurement, obj.curr_confidence)

    def time_float_to_time(self, val): 
        sec = math.floor(val)
        ns = val - sec
        return Time(seconds=sec, nanoseconds=ns)
                            
    def detect_inter_robot_associations(self): 
        '''Detect inter-robot object associations: runs every self.params['sort.re_cluster_secs'] 
        seconds from associations_ros_driver.py
        '''
        
        # self.single_agent_tests()
        
        # neighbors_is_in_range: bool --> in range of any others?
        neighbors_is_in_range, neighbors_in_range_list = self.neighbor_manager.check_neighbors_in_range()
        
        # neighbors_in_range_list will always have one element (same robot_id)
        if len(neighbors_in_range_list) > 1 and self.neighbor_manager.local_robot_is_broker():
            print(f"\nClustering from robot {self.params['robot_id']}: neighbors_is_in_range: {neighbors_is_in_range}\tneighbors_in_range_list: {neighbors_in_range_list}")
            self.cluster(set(neighbors_in_range_list))
            
            self.unify_clustered_objects(neighbors_in_range_list)

            print("\n")

    def single_agent_tests(self): 
        if len(self.unified.get_parents_keys()) == 0: 
            print(f"No detections?")
        else: 
            first_key = self.unified.get_parents_keys()[0]
            first_key_info = self.unified.obj_desc[first_key]
            
            self.single_agent_location_test(first_key_info.time, first_key_info.frame_id, 
                                            first_key_info.kalman_filter)

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
                if j not in self.unified.obj_desc: # clustered already 
                    continue

                j_info = self.unified.obj_desc[j]
                
                if j_info.time > latest_msg_time: 
                    latest_msg_time = j_info.time

                # One place where the ordered dict by insertion part is necessary
                if j_info.time < self.last_time_clustered: 
                    break
                
                heap = []
                for k in reversed(second): 
                    if k not in self.unified.obj_desc: # clustered already
                        continue
                    
                    k_info = self.unified.obj_desc[k]

                    if k_info.time > latest_msg_time: 
                        latest_msg_time = k_info.time
                                            
                    # Assuming each class is very different from each other class, don't cluster if detected as a 
                    # different class
                    if j_info.robot_id == k_info.robot_id or j_info.class_id != k_info.class_id: 
                        continue
                    
                    # r0_frame_id, r0_id, r0_time, r0_kalman_filter, 
                    # r1_frame_id, r1_id, r1_time, r1_kalman_filter, 
                    check_odom = self.check_same_location(j_info.frame_id, j_info.time, 
                                                          j_info.kalman_filter, 
                                                          k_info.frame_id, k_info.time, 
                                                          k_info.kalman_filter)
                    
                    # Has bug of transform requiring extrapolation into the past --> TODO fix
                    print(f"Check odom: {check_odom}")
                    if not check_odom: 
                        continue
                    
                    # Assume there's only one bounding box around any given object, due to NMS working perfectly
                    check_feature_desc = self.check_features(j_info.feature_desc, k_info.feature_desc)
                    
                    if check_feature_desc > self.params['sort.compact_desc_min_similarity']: 
                        heapq.heappush(heap, (-1 * check_feature_desc, k_info.obj_id)) # heapq is a min heap, NOT a max heap
            
                if len(heap) != 0: 
                    _, closest_obj_id = heapq.heappop(heap)
                    self.unified.union(j, f"{i + 1}.{closest_obj_id}")

        self.last_time_clustered = latest_msg_time
        
    def single_agent_location_test(self, r0_time, r0_frame_id, kf): 
        if 0 not in self.last_time_agent_entered: 
            print(f"0 not detected?")
        else: 
            try: 
                A_now_time = self.time_float_to_time(self.last_time_agent_entered[0])
                
                r0_when = self.time_float_to_time(r0_time)
                
                A_now_to_r0_frame_then = self.tf_buffer.lookup_transform_full(
                    "A_odom", A_now_time, 
                    r0_frame_id, r0_when, 
                    "world", Duration(seconds=0.05))
                
                print(f"A_now_to_r0_frame_then: {A_now_to_r0_frame_then}\n")
                
                r0_loc = Vector3Stamped(vector=Vector3(x=kf.x[0], y=kf.x[1], z=kf.x[2]))
                print(f"r0_loc: {r0_loc}\n")
                
                r0_loc_trans = do_transform_vector3(r0_loc, A_now_to_r0_frame_then)
                print(f"r0_loc_trans: {r0_loc_trans}\n")
                
                r0_x = A_now_to_r0_frame_then.transform.translation.x + r0_loc_trans.vector.x
                r0_y = A_now_to_r0_frame_then.transform.translation.y + r0_loc_trans.vector.y
                r0_z = A_now_to_r0_frame_then.transform.translation.z + r0_loc_trans.vector.z
                
                r0_obj_loc = np.array([r0_x, r0_y, r0_z])
                
                print(f"Location of object: {r0_obj_loc}\n\n")

            except Exception as e: 
                print(f"Exception with transformation: {e}")
                traceback.print_exc()


    # Blocking duration is arbitrarily chosen to reduce chance of getting extrapolation into the future 
    # errors, can be changed
    def check_same_location(self, 
                            r0_frame_id, r0_time, r0_kf, 
                            r1_frame_id, r1_time, r1_kf): 
        '''Check if location of object is same between two robots and two timestamps\n
        Convention is that translations are applied before rotations in Transform messages\n
        NOTE: won't scale if A is not present in range
        '''
        try: 
            id = self.params['robot_id']
            last_time_entered = self.last_time_agent_entered[id]
            broker_now_time = self.time_float_to_time(last_time_entered)

            r0_when = self.time_float_to_time(r0_time)
            
            broker_now_to_r0_frame_then = self.tf_buffer.lookup_transform_full(
                self.frame_ids[id], broker_now_time, # was Time()
                r0_frame_id, r0_when, 
                "world", Duration(seconds=0.5))
            
            print(f"broker_now_to_r0_frame_then: {broker_now_to_r0_frame_then}\n")
            
            r0_loc = Vector3Stamped(vector=Vector3(x=r0_kf.x[0], y=r0_kf.x[1], z=r0_kf.x[2]))
            print(f"r0_loc: {r0_loc}\n")

            r0_loc_trans = do_transform_vector3(r0_loc, broker_now_to_r0_frame_then)
            print(f"r0_loc_trans: {r0_loc_trans}\n")

            # Object pose
            r0_x = broker_now_to_r0_frame_then.transform.translation.x + r0_loc_trans.vector.x
            r0_y = broker_now_to_r0_frame_then.transform.translation.y + r0_loc_trans.vector.y
            r0_z = broker_now_to_r0_frame_then.transform.translation.z + r0_loc_trans.vector.z
            
            r0_obj_loc = np.array([r0_x, r0_y, r0_z])
            r0_cov = r0_kf.P[:3, :3]
            
            # --------------r1-----------------

            r1_when = self.time_float_to_time(r1_time)
            
            broker_now_to_r1_frame_then = self.tf_buffer.lookup_transform_full(
                self.frame_ids[id], broker_now_time, # was Time() 
                r1_frame_id, r1_when, 
                "world", Duration(seconds=0.5))
            
            print(f"broker_now_to_r1_frame_then: {broker_now_to_r1_frame_then}\n")
            

            r1_loc = Vector3Stamped(vector=Vector3(x=r1_kf.x[0], y=r1_kf.x[1], z=r1_kf.x[2]))
            print(f"r1_loc: {r1_loc}\n")

            r1_loc_trans = do_transform_vector3(r1_loc, broker_now_to_r1_frame_then)
            print(f"r1_loc_trans: {r1_loc_trans}\n")

            # Object pose
            r1_x = broker_now_to_r1_frame_then.transform.translation.x + r1_loc_trans.vector.x
            r1_y = broker_now_to_r1_frame_then.transform.translation.y + r1_loc_trans.vector.y
            r1_z = broker_now_to_r1_frame_then.transform.translation.z + r1_loc_trans.vector.z
            
            r1_obj_loc = np.array([r1_x, r1_y, r1_z])
            r1_cov = r1_kf.P[:3, :3]
            
            
            is_valid = 0
            if self.params['sort.location_dist_metric'] == "bhattacharyya": 
                dist = gaussian_bhattacharyya(r0_obj_loc, r0_cov, r1_obj_loc, r1_cov, True)
                
                print(f"""Distance between obj from r0 ({r0_obj_loc}) and 
                    r1 ({r1_obj_loc}) using Bhattacharyya bound: {dist}\n\n\n""")
                
                is_valid = 1 if dist < self.params['sort.bhattacharyya_location_epsilon'] else 0
            elif self.params['sort.location_dist_metric'] == "euclidean":
                dist = distance.euclidean(r0_obj_loc, r1_obj_loc)
                
                print(f"""Distance between obj from r0 ({r0_obj_loc}) and 
                    r1 ({r1_obj_loc}) using Euclidean distance: {dist}\n\n\n""")
                
                is_valid = 1 if dist < self.params['sort.euclidean_location_epsilon'] else 0
            else: 
                raise Exception("""Location distance metric isn't recognized: 
                                must be \'bhattacharyya\' or \'euclidean\'""")

            return True if is_valid == 1 else False


        except Exception as e: 
            print(f"Exception with transformation: {e}")
               
    def check_features(self, r0_descriptor, r1_descriptor): 
        '''Get cosine similarity between the two descriptors
        '''
        sim = np.dot(r0_descriptor, r1_descriptor) / (norm(r0_descriptor) * norm(r1_descriptor))
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