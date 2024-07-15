class DisjointSetAssociations: 
    '''Uses Disjoint Set data structure to efficiently cluster the detections\n
    Applies path compression and union by rank for amortized O(a(n)), where a(n) is 
    inverse Ackermann function (minimal)
    '''
    # Key is string
    def __init__(self): 
        self.rank = {} # value: int
        self.parent = {} # value: string
        self.time = {} # value: stamp 
        self.location = {} # value: Point
        self.location_persp = {} # value: robot_id (int)
        self.feat_desc = {} # value: 64 element feature descriptor

    # Key is string
    def __init__(self, mot_global_desc_arr): 
        for obj in mot_global_desc_arr: 
            key = f"{obj.robot_id}.{obj.obj_id}"
            self.rank.update({key: 1})
            self.parent.update({key: key})
            self.time.update({key: (obj.header.stamp.sec + obj.header.stamp.nsec)})
            # self.location.update({key: })
            # TODO finish this
    
    # Key is string
    def __init__(self, name_arr, time_arr, location_arr, location_perspective_arr, feat_desc_arr): 
        # Constructor to create and initialize sets of n items 
        self.rank = {f'{x}': 1 for _, x in enumerate(name_arr)} # value: int
        self.parent = {f'{x}': x for _, x in enumerate(name_arr)} # value: string
        self.time = {f'{x}': x for _, x in enumerate(time_arr)} # value: stamp 
        self.location = {f'{x}': x for _, x in enumerate(location_arr)} # value: Point
        self.location_persp = {f'{x}': x for _, x in enumerate(location_perspective_arr)} # value: robot_id (int)
        self.feat_desc = {f'{x}': x for _, x in enumerate(feat_desc_arr)} # value: 64 element feature descriptor
        
    # increased_arr is an array of strings
    def insert_arr(self, increased_arr): 
        self.rank.update({f'{x}': 1 for _, x in enumerate(increased_arr)})
        self.parent.update({f'{x}': x for _, x in enumerate(increased_arr)})
  
    # Finds set of given item x 
    def find(self, x): 
          
        # Finds the representative of the set that x is an element of 
        if (self.parent[x] != x): 
              
            # if x is not the parent of itself 
            # Then x is not the representative of 
            # its set, 
            self.parent[x] = self.find(self.parent[x]) 
              
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
  
        elif self.rank[xset] > self.rank[yset]: 
            self.parent[yset] = xset 
  
        # If ranks are same, then move y under x (doesn't 
        # matter which one goes where) and increment rank of x's tree 
        else: 
            self.parent[yset] = xset 
            self.rank[xset] = self.rank[xset] + 1