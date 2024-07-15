class DisjointSetAssociations: 
    '''Uses Disjoint Set data structure to efficiently cluster the detections\n
    Applies path compression and union by rank for amortized O(a(n)), where a(n) is 
    inverse Ackermann function (minimal)
    '''
    # arr is an array of strings
    def __init__(self, arr): 
        # Constructor to create and initialize sets of n items 
        self.rank = {f'{x}': 1 for _, x in enumerate(arr)}
        self.parent = {f'{x}': x for _, x in enumerate(arr)}
        self.time = 0
        
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