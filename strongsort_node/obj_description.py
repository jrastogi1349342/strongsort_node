class ObjectDescription(): 
    # Future work: replace dist value with probability distribution (ie mean, variance)
    def __init__(self):
        self.dist = -1 # phi
        self.pitch = -1 # rho
        self.yaw = -1 # theta
        self.time = -1 # value: stamp.sec + stamp.nanosec
        self.agent_perspective = -1
        self.descriptor_conf = -1
        self.feature_desc = []
        self.class_id = -1
        self.obj_id = -1
        self.children = []
        
    def __init__(self, dist, pitch, yaw, time, agent_perspective, 
                 descriptor_conf, feature_desc, class_id, obj_id, children): 
        self.dist = dist # phi
        self.pitch = pitch # rho
        self.yaw = yaw # theta
        self.time = time # value: stamp.sec + stamp.nanosec
        self.agent_perspective = agent_perspective
        self.descriptor_conf = descriptor_conf
        self.feature_desc = feature_desc
        self.class_id = class_id
        self.obj_id = obj_id
        self.children = children
        
    def get_time(self): 
        return self.time
        
    def set_time(self, new_time): 
        self.time = new_time