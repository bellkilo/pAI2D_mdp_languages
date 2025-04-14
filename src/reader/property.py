class Property(object):
    def __init__(self, name, criterion, reward):
        self._name = name
        self._criterion = criterion
        self._reward = reward
    
    @property
    def name(self):
        return self._name
    
    @property
    def criterion(self):
        return self._criterion
    
    @property
    def reward(self):
        return self._reward
