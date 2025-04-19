class Property(object):
    def __init__(self, name, criterion, resolutionModel, terminalStateExpression, rewardExpression):
        self._name = name
        self._criterion = criterion
        self._resolutionModel = resolutionModel
        self._terminalStateExpression = terminalStateExpression
        self._rewardExpression = rewardExpression
    
    @property
    def name(self):
        return self._name
    
    @property
    def terminalStateExpression(self):
        return self._terminalStateExpression
    
    @property
    def criterion(self):
        return self._criterion
    
    @property
    def resolutionModel(self):
        return self._resolutionModel
    
    @property
    def rewardExpression(self):
        return self._rewardExpression
