class Function(object):
    def __init__(self, name, type, parameters, body):
        self._name = name
        self._type = type
        self._parameters = parameters
        self._body = body
    
    @property
    def name(self):
        return self._name
    
    @property
    def type(self):
        return self._type
    
    @property
    def argsSize(self):
        return len(self._parameters)

    def toJaniRRepresentation(self):
        return {
            "name": self._name,
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