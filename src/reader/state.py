from copy import deepcopy

class State(object):
    __slots__ = ["_transientVars", "_nonTransientVars", "_setOfLocation", "_hashRepr"]
    def __init__(self, transientVars, nonTransientVars, setOfLocation):
        self._transientVars = transientVars
        self._nonTransientVars = nonTransientVars
        self._setOfLocation = frozenset(setOfLocation)
        self._hashRepr = (frozenset(self._nonTransientVars.values()), self._setOfLocation)

    @property
    def setOfLocation(self):
        return self._setOfLocation
    
    def clone(self, src, dest, assgns, locs, funcGetter):
        """Clone a new state based on the given parameters."""
        assert len(src) == len(dest)

        transientVars = deepcopy(self._transientVars)
        nonTransientVars = deepcopy(self._nonTransientVars)
        
        # Reset all transient variables to their initial values.
        for transientVar in transientVars.values():
            transientVar.resetToInitValue()
        # Apply assignments to non-transient variables.
        for ref, value in assgns.items():
            if ref not in nonTransientVars:
                continue
            nonTransientVars[ref].setValueTo(value.eval(self, funcGetter))
        setOfLocation = self._setOfLocation.difference(src).union(dest)
        # Apply transient value assignments to transient variables.
        for loc in setOfLocation:
            if loc not in locs:
                continue
            for ref, value in locs[loc].items():
                transientVars[ref].setValueTo(value.eval(self, funcGetter))
        return State(transientVars, nonTransientVars, setOfLocation)
    
    def get(self, name, default = None):
        """Return the value of associated variable (both transient or non-transient)
        if it exists, otherwise return the default value."""
        variable = self._transientVars.get(name)
        if variable is not None:
            return variable.value
        variable = self._nonTransientVars.get(name)
        if variable is not None:
            return variable.value
        return default
    
    def getLowMemRepr(self):
        """Return a string representation of the state (low memory representation)."""
        nonTransientVars, setOfLocation = self._hashRepr
        return str((sorted(nonTransientVars, key = lambda x: x.name),
                    sorted(setOfLocation)))
    
    def __contains__(self, item):
        return item in self._transientVars or item in self._nonTransientVars
    
    def __eq__(self, value):
        return isinstance(value, State) and self._hashRepr == value._hashRepr
    
    def __hash__(self):
        return hash(self._hashRepr)
