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
        self.agents_seen = {} # value: set of agents this accounts for 

    # Key is string
    def __init__(self, mot_global_desc_arr): 
        self.insert_arr(mot_global_desc_arr)

    def insert(self, obj, key): 
        self.rank.update({key: 1})
        self.parent.update({key: key})
        self.obj_desc.update({key: ObjectDescription(
            dist=obj.distance, 
            pitch=obj.pitch, 
            yaw=obj.yaw, 
            time=obj.header.stamp.sec + (obj.header.stamp.nanosec / 1000000000), 
            agent_perspective=obj.robot_id, 
            feature_desc=obj.best_descriptor, 
            class_id=obj.obj_class_id
        )})
        self.agents_seen.update({key: set(obj.robot_id)})
        
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
            self.agents_seen[xset].append(self.agents_seen[yset])
            self.obj_desc[xset].children.append(yset).append(self.obj_desc[yset].children)
  
        elif self.rank[xset] > self.rank[yset]: 
            self.parent[yset] = xset 
            self.agents_seen[yset].append(self.agents_seen[xset])
            self.obj_desc[yset].children.append(xset).append(self.obj_desc[xset].children)
  
        # If ranks are same, then move y under x (doesn't 
        # matter which one goes where) and increment rank of x's tree 
        else: 
            self.parent[yset] = xset 
            self.rank[xset] = self.rank[xset] + 1
            self.agents_seen[yset].append(self.agents_seen[xset])
            self.obj_desc[yset].children.append(xset).append(self.obj_desc[xset].children)
                
    def delete(self, key): 
        del self.rank[key], self.parent[key], self.obj_desc[key], self.agents_seen[key]