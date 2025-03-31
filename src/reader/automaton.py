class EdgeDestination(object):
    def __init__(self, destination, probability, assignments, reward):
        self.destination = frozenset(destination)
        self.probability = probability
        self.assignments = assignments

        self.reward = reward

class Edge(object):
    def __init__(self, source, action, guard, edgeDestinations):
        self.source = frozenset(source)
        self.action = action
        self.guard = guard
        self.edgeDestinations = edgeDestinations

    def isSatisfied(self, state):
        if len(self.source) == 1:
            if not state.location & self.source:
                return False
        elif not state.location == self.source:
            return False
        return self.guard.eval(state)
    
    def __repr__(self):
        return f"{self.source} {self.guard} {self.action}"
    
class Automaton(object):
    def __init__(self, name, locations, initLocations, edges):
        self.name = name
        self.locations = locations
        self.initLocations = initLocations
        self.edges = edges

    def __repr__(self):
        return f"{self.name}{self.locations}{self.initLocations}"