class DisjointSetAssociations: 
    # arr is an array of strings
    def __init__(self, arr): 
        # Constructor to create and initialize sets of n items 
        self.rank = [1] * len(arr) 
        self.parent = [i for i in arr] 
        
    # increased_arr is an array of strings
    def insert(self, increased_arr): 
        old_size = len(self.rank)
        self.rank.extend([1] * len(increased_arr))
        self.parent.extend([i for i in increased_arr])
  
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
          
        # Find current sets of x and y 
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