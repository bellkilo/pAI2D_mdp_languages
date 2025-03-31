class Type(object):
    def __init__(self, type, boundaries=None):
        self.type = type
        self.boundaries = boundaries
    
    def isBoundedConsistent(self, value):
        return self.boundaries is None or \
            self.boundaries[0] <= value <= self.boundaries[1]
    
    def __eq__(self, value: 'Type'):
        return isinstance(value, Type) and \
            self.type == value.type and self.boundaries == value.boundaries
    
    def __hash__(self):
        return hash((self.type, self.boundaries))
    
    def __str__(self):
        if self.boundaries is None:
            return f"({self.type}, unbounded)"
        return f"({self.type}, range={self.boundaries})"
    
    def __repr__(self):
        return self.__str__()

class Constant(object):
    def __init__(self, name, type, value):
        self.name = name
        self.type = type
        self.value = value
    
    def __str__(self):
        return f"{self.name}={self.value}"
    
    def __repr__(self):
        return f"Constant(name={self.name}, type={self.type}, value={self.value})"

class Variable(object):
    def __init__(self, name, type: Type, scope, initValue, transient=False):
        self.name = name
        self.type = type
        self.scope = scope
        self.initValue = initValue
        self.transient = transient

        self.__value = self.initValue

    def resetToInitValue(self):
        self.__value = self.initValue 
    
    def setValueTo(self, value):
        if self.type.isBoundedConsistent(value):
            self.__value = value

    def getValue(self):
        return self.__value
    
    def isGlobal(self):
        return self.scope[0] == "global"
    
    def __eq__(self, value: 'Variable'):
        return isinstance(value, Variable) and \
            self.name == value.name and self.type == value.type and \
            self.scope == value.scope and self.transient == value.transient and \
            self.__value == value.__value
    
    def __hash__(self):
        return hash((self.name, self.type, self.scope, self.transient, self.__value))

    def __str__(self):
        if self.scope[0] == "global":
            return f"Global variable '{self.name}': {self.__value}"
        return f"Local variable '{self.name}': {self.__value} (in automaton '{self.scope[1]}')"
    
    def __repr__(self):
        return f"Variable(name={self.name}, type={self.type}, scope={self.scope}, value={self.__value}, transient={self.transient})"
