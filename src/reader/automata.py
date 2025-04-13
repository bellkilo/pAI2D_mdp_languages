class EdgeDestination(object):
    __slots__ = ["_dest", "_prob", "_assgns", "_reward"]
    def __init__(self, dest, prob, assgns, reward):
        self._dest = frozenset(dest)
        self._prob = prob
        self._assgns = assgns
        self._reward = reward

    @property
    def destination(self):
        return self._dest
    
    @property
    def probability(self):
        return self._prob

    @property
    def assignments(self):
        return self._assgns
    
    @property
    def reward(self):
        return self._reward
    
    def toJaniRRepresentation(self):
        """Return the Jani-R representation of an edge destination."""
        return {
            "location": next(iter(self.destination)).split(".")[0],
            "probability": { "exp": self.probability.toJaniRRepresentation() },
            # "reward": { "exp": self._reward.toJaniRRepresentation() },
            "reward": {
                "exp": {
                    "op": "ite",
                    "if": "finished",
                    "then": 0.0,
                    "else": "steps"
                }
                # "exp": {

                # "op": "∧",
                # "left":           {"op": "=", "left": "var7", "right": 1},
                # "right": {"op": "=", "left": "var6", "right": 2}

                # }
            },
            "assignments": [
                {
                    "ref": ref.split(".")[0],
                    "value": value.toJaniRRepresentation()
                }
                for ref, value in self.assignments.items()
            ]
        }
    
class Edge(object):
    __slots__ = ["_src", "_action", "_guard", "_edgeDests"]
    def __init__(self, src, action, guard, edgeDests):
        self._src = frozenset(src)
        self._action = action
        self._guard = guard
        self._edgeDests = edgeDests
    
    @property
    def source(self):
        return self._src
    
    @property
    def action(self):
        return self._action
    
    @property
    def guard(self):
        return self._guard
    
    @property
    def edgeDestinations(self):
        return self._edgeDests
    
    def isSatisfied(self, state, funcGetter):
        """Return True if the guard is satisfied, otherwise return False."""
        return state.setOfLocation.issuperset(self.source) and self.guard.eval(state, funcGetter)
    
    def toJaniRRepresentation(self):
        """Return the Jani-R representation of an edge."""
        janiRRepr = {
            "location": next(iter(self.source)).split(".")[0],
            "guard": { "exp": self.guard.toJaniRRepresentation() },
            "destinations": [ edgeDest.toJaniRRepresentation() for edgeDest in self.edgeDestinations ]
        }
        if self.action != "silentAction":
            janiRRepr["action"] = self.action
        return janiRRepr

class Automata(object):
    __slots__ = ["_name", "_locs", "_initLoc", "_edges"]
    def __init__(self, name, locs, initLoc, edges):
        self._name = name
        self._locs = locs
        self._initLoc = frozenset(initLoc)
        self._edges = edges
    
    @property
    def name(self):
        return self._name
    
    @property
    def locations(self):
        return self._locs
    
    @property
    def initLocation(self):
        return self._initLoc
    
    @property
    def edges(self):
        return self._edges
    
    def toJaniRRepresentation(self):
        """Return the Jani-R representation of an automaton."""
        return {
            "name": self.name,
            "initial-locations": [ loc.split(".")[0] for loc in self.initLocation ],
            "locations": [
                {
                    "name": name.split(".")[0],
                    "transient-values": [
                        { "ref": ref.split(".")[0], "value": value.toJaniRRepresentation() }
                        for ref, value in transientValues.items()
                    ]
                }
                for name, transientValues in self.locations.items()
            ],
            "edges": [ edge.toJaniRRepresentation() for edge in self.edges ]
        }
