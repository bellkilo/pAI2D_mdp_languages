class Function(object):
    def __init__(self, name, type, scope, parameters, body):
        self._name = name
        self._type = type
        self._scope = scope
        self._parameters = parameters
        self._body = body
    
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
    def parameters(self):
        return self._parameters

    @property
    def body(self):
        return self._body
    
    def isGlobal(self):
        return self.scope[0] == "global"

    def toJaniRRepresentation(self):
        return {
            "name": self._name.split(".")[0],
            "type": self._type.toJaniRRepresentation(),
            "parameters": [
                {
                    "name": name,
                    "type": type.toJaniRRepresentation()
                }
                for name, type in self._parameters.items()
            ],
            "body": self._body.toJaniRRepresentation()
        }