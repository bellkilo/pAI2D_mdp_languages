from variable import Constant, Variable
from automaton import Automaton, Edge, EdgeDestination
from expression import Expression

from typing import Dict, Set, Tuple

from copy import deepcopy
from itertools import product

class DuplicateActionError(KeyError):
    pass

class DuplicateConstantError(KeyError):
    pass

class DuplicateVariableError(KeyError):
    pass

########### Does not modify ###########
class Model(object):
    __slots__ = ("name", "type", "__actions", "__constants", "__transientVars", "__nonTransientVars",
                "__automaToIndex", "__syncActionsList", "__nonSyncAutomas", "__syncAutoma")
    
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

        self.__actions = set()
        self.__constants = dict()

        self.__transientVars = dict()
        self.__nonTransientVars = dict()
    
    def addAction(self, action: str):
        """Add an action to the model."""
        if action in self.__actions:
            raise DuplicateActionError(f"Action '{action}' already exists in the model '{self.name}'")
        self.__actions.add(action)
    
    def containsAction(self, action: str):
        """Return True if the model contains the action, otherwise False."""
        return action in self.__actions
    
    def addConstant(self, constant: Constant):
        """Add a constant to the model."""
        name = constant.name
        if name in self.__constants:
            raise DuplicateConstantError(f"Constant '{name}' already exists in the model '{self.name}'")
        self.__constants[name] = constant

    def isConstant(self, name: str):
        """Return True if 'name' is declared as a constant variable in the model."""
        return name in self.__constants
    
    def getConstantValue(self, name: str):
        """Return the value associated with a constant variable."""
        constant = self.__constants.get(name)
        if constant is not None:
            return constant.value
        raise KeyError(f"Unrecognized constant name '{name}' for model '{self.name}'")

    def addVariable(self, variable: Variable):
        """Add a variable to the model."""
        name = variable.name
        if variable.transient:
            if name in self.__transientVars:
                raise DuplicateVariableError(f"Transient variable '{name}' already exists in the model '{self.name}'")
            self.__transientVars[name] = variable
        else:
            if name in self.__nonTransientVars:
                raise DuplicateVariableError(f"Non transient variable '{name}' already exists in the model '{self.name}'")
            self.__nonTransientVars[name] = variable
    
    def isGlobal(self, name: str):
        """Return True if 'name' is declared as a costant or global variable (both transient or non-transient)."""
        if self.isConstant(name):
            return True
        var = self.__nonTransientVars.get(name)
        if var is not None:
            return var.isGlobal()
        var = self.__transientVars.get(name)
        if var is not None:
            return var.isGlobal()
        return False
    
    def isTransientVariable(self, name: str):
        """Return True if 'name' is declared as a transient variable, otherwise False."""
        return name in self.__transientVars
    
    def getInitState(self):
        """Return the initial state."""
        return State(self.__transientVars,
                     self.__nonTransientVars,
                     self.__syncAutoma.initLocations)

    def setSystemInformations(self, automaToIndex, syncActionsList):
        """Add system informations, in particular, automaton informations."""
        self.__automaToIndex = automaToIndex
        self.__syncActionsList = syncActionsList
        self.__nonSyncAutomas = [None] * len(self.__automaToIndex)

    def addAutomaton(self, automaton: Automaton):
        """Add a non-synchronized automaton to the model."""
        name = automaton.name
        if name not in self.__automaToIndex:
            raise KeyError(f"Unrecognized automaton '{name}' for model '{self.name}'")
        index = self.__automaToIndex[name]
        self.__nonSyncAutomas[index] = automaton
    
    def synchronize(self):
        self.__syncAutoma = self.__synchronizeAutomaton()
        del self.__nonSyncAutomas

    def __synchronizeAutomaton(self) -> Automaton:
        if len(self.__automaToIndex) == 1:
            return self.__nonSyncAutomas[0]
        
        syncNonSilentActions = set()
        syncEdges = list()
        syncInitLocations = set()
        syncLocations = dict()

        actionMapList = []
        nonSyncAutomas = self.__nonSyncAutomas
        for automa in nonSyncAutomas:
            syncLocations.update(automa.locations)
            syncInitLocations.update(automa.initLocations)
            actionMapEdges = dict()

            for edge in automa.edges:
                action = edge.action
                if action == "silentAction":
                    syncEdges.append(edge)
                else:
                    actionMapEdges.setdefault(action, []).append(edge)
            actionMapList.append(actionMapEdges)
        
        for result, syncActions in self.__syncActionsList:
            syncNonSilentActions.add(result)

            preSyncEdgesList = []
            for i, syncAction in enumerate(syncActions):
                if syncAction is not None:
                    preSyncEdgesList.append(actionMapList[i][syncAction])


            if len(preSyncEdgesList) == 1:
                for edge in preSyncEdgesList[0]:
                    syncEdges.append(edge)
                continue

            for edgeComb in product(*preSyncEdgesList):
                syncEdges.append(self.__synchronizeEdge(edgeComb, result))
            
        self.__actions = syncNonSilentActions
        return Automaton("Main", syncLocations, syncInitLocations, syncEdges)
    
    def __synchronizeEdge(self, edgeComb: Tuple[Edge], action: str) -> Edge:
        syncSource = set()
        syncGuard = Expression("bool", True)


        preSyncEdgesDestinationsList = []
        for edge in edgeComb:
            syncGuard = Expression.mergeExpression(syncGuard, edge.guard)
            syncSource.update(edge.source)

            preSyncEdgesDestinationsList.append(edge.edgeDestinations)
        
        syncEdgeDestinations = [ self.__synchronizeEdgeDestination(edgeDestComb) 
                                 for edgeDestComb in product(*preSyncEdgesDestinationsList) ]
        return Edge(syncSource, action, syncGuard, syncEdgeDestinations)
    
    def __synchronizeEdgeDestination(self, edgeDestComb: Tuple[EdgeDestination]) -> EdgeDestination:
        syncProb = Expression("real", 1.)
        syncDest = set()
        syncAssigns = dict()
        syncReward = Expression("int", 0)

        for edgeDest in edgeDestComb:
            syncProb = Expression.mergeExpression(syncProb, edgeDest.probability)
            syncDest.update(edgeDest.destination)
            syncAssigns.update(edgeDest.assignments)
            syncReward = Expression.reduceExpr("+", syncReward, edgeDest.reward)
        return EdgeDestination(syncDest, syncProb, syncAssigns, syncReward)
    
    def getSyncAutoma(self):
        return self.__syncAutoma
    
    def getActions(self):
        return self.__actions

#######################################

class State(object):
    __slots__ = ("__transientVars", "__nonTransientVars", "__location", "__hashObject")

    def __init__(self,
                 transientVars: Dict[str, Variable],
                 nonTransientVars: Dict[str, Variable],
                 location: Set[str]):
        self.__transientVars = transientVars
        self.__nonTransientVars = nonTransientVars
        self.__location = frozenset(location)

        self.__hashObject = (frozenset(self.__nonTransientVars.values()), self.__location)
    
    def clone(self,
              source: Set[str],
              destination: Set[str],
              assignments: Dict[str, Expression]):
        transientVars = deepcopy(self.__transientVars)
        nonTransientVars = deepcopy(self.__nonTransientVars)

        # for var in transientVars.values():
        #     var.resetToInitValue()

        for ref, expr in assignments.items():
            nonTransientVars[ref].setValueTo(expr.eval(self))

        location = self.__location.difference(source).union(destination)

        return State(transientVars, nonTransientVars, location)
        
    def get(self, name):
        var = self.__transientVars.get(name)
        if var is not None:
            return var.getValue()
        variable = self.__nonTransientVars.get(name)
        if variable is not None:
            return variable.getValue()
        raise KeyError(f"Unrecognized key '{name}' for state")
    
    @property
    def location(self):
        return self.__location
    
    def __eq__(self, value: 'State'):
        return isinstance(value, State) and self.__hashObject == value.__hashObject
    
    def __hash__(self):
        return hash(self.__hashObject)