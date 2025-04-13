class Type(object):
    __slots__ = ["_type", "_boundaries"]

    def __init__(self, type, boundaries = None):
        self._type = type
        self._boundaries = boundaries
    
    @property
    def type(self):
        return self._type
    
    @property
    def boundaries(self):
        return self._boundaries

    def isBoundedConsistent(self, value):
        """Return True if the boundaries are consistent, otherwise return False."""
        return self.boundaries is None or (self.boundaries[0] <= value <= self.boundaries[1])
    
    def __eq__(self, value):
        return isinstance(value, Type) and self.type == value.type and self.boundaries == value.boundaries
    
    def __hash__(self):
        return hash((self.type, self.boundaries))
    
    def toJaniRRepresentation(self):
        """Return the Jani-R representation of an type."""
        if self.boundaries is not None:
            return {
                "kind": "bounded",
                "base": self.type,
                "lower-bound": self.boundaries[0],
                "upper-bound": self.boundaries[1]
            }
        return self.type

    
class Constant(object):
    def __init__(self, name, type, value):
        self._name = name
        self._type = type
        self._value = value
    
    @property
    def name(self):
        return self._name
       
    @property
    def type(self):
        return self._type
     
    @property
    def value(self):
        return self._value

    def toJaniRRepresentation(self):
        """Return the Jani-R representation of a constant variable."""
        return {
            "name": self.name,
            "type": self.type.toJaniRRepresentation(),
            "value": self.value
        }


class Variable(object):
    __slots__ = ["_name", "_type", "_scope", "_initValue", "_value", "_transient"]

    def __init__(self, name, type, scope, initValue = None, transient = False):
        self._name = name
        self._type = type
        self._scope = scope
        self._initValue = initValue
        self._value = self._initValue
        self._transient = transient
    
    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type
    
    @property
    def scope(self):
        return self._scope
    
    @property
    def value(self):
        return self._value

    @property
    def transient(self):
        return self._transient

    def isGlobal(self):
        """Return True if it is declared as a global variable, otherwise return False."""
        return self.scope[0] == "global"

    def resetToInitValue(self):
        """Reset the variable to its initial value."""
        if self._initValue is None:
            raise Exception(f"Variable '{self.name}' has no initial value")
        self._value = self._initValue

    def setValueTo(self, value):
        """Set the variable value to."""
        if self.type.isBoundedConsistent(value):
            self._value = value
    
    def instantiate(self):
        """Return a list of variables instantiated with values."""
        if self._initValue is not None:
            return [self._clone(self._initValue)]
        low, up = self._type.boundaries
        return list(map(self._clone, range(low, up + 1)))
    
    def _clone(self, value):
        return Variable(self._name, self._type, self._scope, value, self._transient)
    
    def __eq__(self, value):
        return isinstance(value, Variable) and \
            self.name == value.name and self.type == value.type and \
            self.scope == value.scope and self.transient == value.transient and \
            self.value == value.value
    
    def __hash__(self):
        return hash((self.name, self.type, self.scope, self.transient, self.value))
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return f"{self.name}={self.value}"
    
    def toJaniRRepresentation(self):
        """Return the Jani-R representation of a variable."""
        janiRRepr =  {
            "name": self.name.split(".")[0],
            "type": self.type.toJaniRRepresentation(),
            "transient": self.transient
        }
        if self._initValue is not None:
            janiRRepr["initial-value"] = self._initValue
        return janiRRepr
