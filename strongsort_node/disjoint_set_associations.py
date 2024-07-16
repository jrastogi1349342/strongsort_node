from strongsort_node.obj_description import ObjectDescription

class DisjointSetAssociations: 
    '''Uses Disjoint Set data structure to efficiently cluster the detections\n
    Applies path compression and union by rank for amortized O(a(n)), where a(n) is 
    inverse Ackermann function (minimal)
    '''
    # Key is string
    def __init__(self, num_robots): 
        self.num_robots = num_robots
        # This assumes python >= 3.6, where dicts are ordered by order of insertion
        self.rank = {} # value: int
        self.parent = {} # value: string
        self.obj_desc = {} # value: ObjectDescription 
        self.is_parent = {}

    # Key is string
    def __init__(self, mot_global_desc_arr): 
        self.insert_arr(mot_global_desc_arr)

    def insert(self, obj, key, curr_time): 
        self.rank.update({key: 1})
        self.parent.update({key: key})
        self.obj_desc.update({key: ObjectDescription(
            dist=obj.distance, 
            pitch=obj.pitch, 
            yaw=obj.yaw, 
            time=curr_time, 
            robot_id=obj.robot_id, 
            descriptor_conf=obj.max_confidence,
            feature_desc=obj.best_descriptor, 
            class_id=obj.obj_class_id, 
            obj_id=obj.obj_id, 
            children={}
        )})
        self.is_parent.update({key: True})
        
    # mot_global_desc_arr is an array of MOTGlobalDescriptor objects
    def insert_arr(self, mot_global_desc_arr): 
        for obj in mot_global_desc_arr: 
            self.insert(obj, f'{obj.robot_id}.{obj.obj_id}')
            
    def find(self, x): 
        # Check if x exists in disjoint set
        if x not in self.parent: 
            return ''
        else: 
            return self.find_helper(x)
  
    # Finds root of set of given item x, which exists in the disjoint set
    def find_helper(self, x):         
        # Finds the representative of the set that x is an element of 
        if (self.parent[x] != x): 
              
            # if x is not the parent of itself 
            # Then x is not the representative of 
            # its set, 
            self.parent[x] = self.find_helper(self.parent[x]) 
              
            # so we recursively call find on its parent and move i's 
            # node directly under the representative of this set 
        return self.parent[x] 
  
    # Do union of two sets represented by x and y. 
    # TODO delete information from child nodes to save memory
    def union(self, x, y): 
          
        # Find current root ancestors of x and y 
        xset = self.find(x) 
        yset = self.find(y) 
  
        # If they are already in same set 
        if xset == yset: 
            return
  
        # Put smaller ranked item under bigger ranked item 
        # if ranks are different 
        if self.rank[xset] < self.rank[yset]: 
            self.parent[xset] = yset 

            self.obj_desc[yset].children.update({self.obj_desc[xset].robot_id: xset})
            for child_id, child_key in self.obj_desc[xset].children.items():
                self.obj_desc[yset].children.update({child_id: child_key})
                
            self.obj_desc[yset].time = self.obj_desc[yset].time if self.obj_desc[yset].time > self.obj_desc[xset].time else self.obj_desc[xset].time
            self.is_parent[xset] = False
  
        elif self.rank[xset] > self.rank[yset]: 
            self.parent[yset] = xset 

            self.obj_desc[xset].children.update({self.obj_desc[yset].robot_id: yset})
            for child_id, child_key in self.obj_desc[yset].children.items():
                self.obj_desc[xset].children.update({child_id: child_key})
                
            self.obj_desc[xset].time = self.obj_desc[xset].time if self.obj_desc[xset].time > self.obj_desc[yset].time else self.obj_desc[yset].time
            self.is_parent[yset] = False
  
        # If ranks are same, then move y under x (doesn't 
        # matter which one goes where) and increment rank of x's tree 
        else: 
            self.parent[yset] = xset 
            self.rank[xset] = self.rank[xset] + 1
            
            self.obj_desc[xset].children.update({self.obj_desc[yset].robot_id: yset})
            for child_id, child_key in self.obj_desc[yset].children.items():
                self.obj_desc[xset].children.update({child_id: child_key})

            self.obj_desc[xset].time = self.obj_desc[xset].time if self.obj_desc[xset].time > self.obj_desc[yset].time else self.obj_desc[yset].time
            self.is_parent[yset] = False
                
    def delete(self, key): 
        del self.rank[key], self.parent[key], self.obj_desc[key], self.is_parent[key]
        
    def get_parents_keys(self): 
        return [key for key, value in self.is_parent.items() if value]
    
    def get_all_clustered_keys(self, parent_key): 
        lst = [parent_key]
        
        for robot_id, obj_id in self.obj_desc[parent_key].children: 
            lst.append(f'{robot_id}.{obj_id}')
            
        return lst
    
    # No negative values in clusters
    def get_obj_id_in_cluster(self, parent_key, robot_id): 
        if f'{robot_id}.' in parent_key: 
            return self.obj_desc[parent_key].robot_id
        else: 
            if robot_id in self.obj_desc[parent_key].children: 
                return self.obj_desc[parent_key].children[robot_id]
            else: 
                return -1