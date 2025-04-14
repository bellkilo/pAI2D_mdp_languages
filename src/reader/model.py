from itertools import product
from collections import deque
from copy import deepcopy

import json
import numpy as np

from automata import Automata, Edge, EdgeDestination
from expression import Expression
from state import State

import marmote.core as mc
import marmote.mdp as mmdp

class _BaseModel(object):
    __slots__ = [
        "_name", "_type", "_actions", "_constants", "_transientVars", "_nonTransientVars",
        "_functions", "_automataMapIndex", "_preSyncActionsList", "_nonSyncAutomatas", "_automata"
    ]

    def __init__(self, name, type):
        self._name = name
        self._type = type
        
        self._actions = set()
        self._constants = dict()

        self._transientVars = dict()
        self._nonTransientVars = dict()

        self._functions = dict()

        self._automataMapIndex = None
        self._preSyncActionsList = None
        self._nonSyncAutomatas = None
        self._automata = None

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    def addAction(self, action):
        """Add an action to the model."""
        if action in self._actions:
            raise KeyError(f"Action '{action}' already exists in model '{self.name}'")
        self._actions.add(action)

    def containsAction(self, action):
        """Return True if the model contains action, otherwise return False."""
        return action in self._actions

    def addConstant(self, constant):
        """Add a constant variable to the model."""
        name = constant.name
        if name in self._constants:
            raise KeyError(f"Constant variable '{name}' already exists in model '{self.name}'")
        self._constants[name] = constant
    
    def isConstantVariable(self, name):
        """Return True if name is declared as a constant variable in the model."""
        return name in self._constants
    
    def getConstantValue(self, name):
        """Return the associated value of a constant variable, otherwise raise a KeyError."""
        constant = self._constants.get(name)
        if constant is not None:
            return constant.value
        raise KeyError(f"Unrecognized constant name '{name}' for model '{self.name}'")
    
    def addVariable(self, variable):
        """Add a variable to the model."""
        name = variable.name
        transient = variable.transient
        if transient:
            if name in self._transientVars:
                raise KeyError(f"Transient variable '{name}' already exists in model '{self.name}'")
            self._transientVars[name] = variable
        else:
            if name in self._nonTransientVars:
                    raise KeyError(f"Non-transient variable '{name}' already exists in model '{self.name}'")
            self._nonTransientVars[name] = variable
    
    def isGlobalVariable(self, name):
        """Return True if the variable name is declared as a constant or global variable."""
        if self.isConstantVariable(name):
            return True
        variable = self._transientVars.get(name)
        if variable is not None:
            return variable.isGlobal()
        variable = self._nonTransientVars.get(name)
        if variable is not None:
            return variable.isGlobal()
        return False
    
    def isLocalVariable(self, name):
        """Return True if the variable name is declared as a local variable."""
        variable = self._transientVars.get(name)
        if variable is not None:
            return not variable.isGlobal()
        variable = self._nonTransientVars.get(name)
        if variable is not None:
            return not variable.isGlobal()
        return False

    def isTransientVariable(self, name):
        """Return True if name is declared as a transient variable."""
        return name in self._transientVars
    
    def containsVariable(self, name):
        """Return True if the model contains variable name, otherwise return False."""
        return name in self._constants or name in self._transientVars or name in self._nonTransientVars

    def declareFunction(self, name):
        """Declare the name as a function."""
        if name in self._functions:
            raise KeyError(f"Function '{name}' already exists in model '{self.name}'")
        self._functions[name] = None

    def addFunction(self, function):
        """Add a function to the model."""
        name = function.name
        if name not in self._functions:
            raise KeyError(f"Undeclared function name '{name}' for model '{self.name}'")
        self._functions[name] = function
    
    def containsFunction(self, name):
        """Return True if the model contains function, otherwise return False."""
        return name in self._functions
    
    def getFunction(self, name):
        """Return the function associated with the name."""
        if not self.containsFunction(name):
            raise SyntaxError(f"Unrecognized function '{name}' for model '{self.name}'")
        return self._functions[name]

    def setSystemInformation(self, automataMapIndex, preSysnActionsList):
        """Add system information to the model."""
        if self._automataMapIndex is not None and self._preSyncActionsList is not None:
            raise Exception("System information has already been defined and cannot be redefined")
        self._automataMapIndex = automataMapIndex
        self._preSyncActionsList = preSysnActionsList
        self._nonSyncAutomatas = [None] * len(self._automataMapIndex)

    def addAutomata(self, automata):
        """Add a non-synchronized automata to the model."""
        if self._automataMapIndex is None:
            raise Exception("System information not defined")
        name = automata.name
        if name not in self._automataMapIndex:
            raise KeyError(f"Unrecognized automata '{name}' for model '{self.name}'")
        idx = self._automataMapIndex[name]
        self._nonSyncAutomatas[idx] = automata
    
    def getInitStates(self):
        """Return all possible initial states."""
        states = []
        locs = self._automata.locations
        setOfLocation = self._automata.initLocation
        for vars in product(*map(lambda x: x.instantiate(), self._nonTransientVars.values())):
            transientVars = deepcopy(self._transientVars)
            nonTransientVars = { var.name: var for var in vars }
            for loc in setOfLocation:
                if loc not in locs:
                    continue
                for ref, value in locs[loc].items():
                    transientVars[ref].setValueTo(value.eval(nonTransientVars, self._functions))
            states.append(State(transientVars, nonTransientVars, setOfLocation))
        return states

    def synchronize(self):
        """Synchronize automata."""
        if self._automata is not None:
            raise Exception("Synchronization is complete")
        self._automata = self._synchronizeAutomata()
        del self._nonSyncAutomatas
    
    def _synchronizeAutomata(self):
        # If there is only 1 automata, then synchronization is not necessary,
        # we simply return this automata directly.
        if len(self._nonSyncAutomatas) == 1:
            return self._nonSyncAutomatas[0]
        
        syncActions = set()
        syncEdges = list()
        syncInitLoc = set()
        syncLocs = dict()

        # Preprocessing.
        actionMapEdgesList = []
        for automata in self._nonSyncAutomatas:
            syncInitLoc.update(automata.initLocation)
            syncLocs.update(automata.locations)

            actionMapEdges = dict()
            for edge in automata.edges:
                action = edge.action
                if action == "silentAction":
                    syncEdges.append(edge)
                else:
                    actionMapEdges.setdefault(action, []).append(edge)
            actionMapEdgesList.append(actionMapEdges)

        # Synchronization.
        for result, preSyncActions in self._preSyncActionsList:
            syncActions.add(result)
            preSyncEdgesList = []
            for i, preSyncAcition in enumerate(preSyncActions):
                if preSyncAcition is not None:
                    preSyncEdgesList.append(actionMapEdgesList[i][preSyncAcition])

            for edgeComb in product(*preSyncEdgesList):
                syncEdges.append(self._synchronizeEdge(edgeComb, result))
        self._actions = syncActions
        return Automata("Main", syncLocs, syncInitLoc, syncEdges)
    
    def _synchronizeEdge(self, edgeComb, action):
        syncSrc = set()
        syncGuard = Expression("bool", True)

        preSyncEdgeDestsList = []
        for edge in edgeComb:
            syncSrc.update(edge.source)
            syncGuard = Expression.reduceExpression("∧", syncGuard, edge.guard)
            preSyncEdgeDestsList.append(edge.edgeDestinations)
        
        syncEdgeDests = [
            self._synchronizeEdgeDestination(edgeDestComb) for edgeDestComb in product(*preSyncEdgeDestsList)
        ]
        return Edge(syncSrc, action, syncGuard, syncEdgeDests)
    
    def _synchronizeEdgeDestination(self, edgeDestComb):
        syncDest = set()
        syncProb = Expression("real", 1.)
        syncAssngs = dict()
        syncReward = Expression("int", 0)

        for edgeDest in edgeDestComb:
            syncDest.update(edgeDest.destination)
            syncProb = Expression.reduceExpression("*", syncProb, edgeDest.probability)
            syncAssngs.update(edgeDest.assignments)
            syncReward = Expression.reduceExpression("+", syncReward, edgeDest.reward) # TODO
        return EdgeDestination(syncDest, syncProb, syncAssngs, syncReward)


class JaniModel(_BaseModel):
    def __init__(self, name, type):
        super().__init__(name, type)

        self._properties = dict()
    
    def addProperty(self, property):
        """Add a property to the model."""
        # TODO
        name = property.name
        self._properties[name] = property
        pass
    
    def writeJaniR(self, path, targetProperty):
        modelStruct = dict()
        modelStruct["name"] = self.name

        # TODO
        modelStruct["type"] = "TotalRewardMDP"
        modelStruct["criterion"] = "max"
        ####

        modelStruct["constants"] = [
            constant.toJaniRRepresentation()
            for constant in self._constants.values()
        ]

        modelStruct["actions"] = [
            { "name": action }
            for action in self._actions
        ]

        globalVars = list()
        localVars = dict()
        for var in { **self._transientVars, **self._nonTransientVars }.values():
            if var.isGlobal():
                globalVars.append(var.toJaniRRepresentation())
            else:
                _, automata = var.scope
                localVars.setdefault(automata, []).append(var.toJaniRRepresentation())
        modelStruct["variables"] = globalVars

        globalFuncs = list()
        localFuncs = dict()
        for func in self._functions.values():
            if func.isGlobal():
                globalFuncs.append(func.toJaniRRepresentation())
            else:
                _, automata = func.scope
                localFuncs.setdefault(automata, []).append(func.toJaniRRepresentation())
        modelStruct["functions"] = globalFuncs

        modelStruct["system"] = {
            "elements": [
                { "automaton": automata }
                for automata in self._automataMapIndex.keys()
            ],
            "syncs": [
                {
                    "synchronise": syncActions,
                    "result": result
                } for result, syncActions in self._preSyncActionsList
            ]
        }

        modelStruct["automata"] = [
            automata.toJaniRRepresentation()
            for automata in self._nonSyncAutomatas
        ]
        for automata in modelStruct["automata"]:
            name = automata["name"]
            if name in localVars:
                automata["variables"] = localVars[name]
            if name in localFuncs:
                automata["functions"] = localFuncs[name]

        # TODO
        if targetProperty is not None:
            property = self._properties[targetProperty]
            reward = property.reward.toJaniRRepresentation()
            for automata in modelStruct["automata"]:
                for edges in automata["edges"]:
                    for edgeDest in edges["destinations"]:
                        edgeDest["reward"]["exp"] = reward
        ####
        
        with open(path, "w", encoding="utf-8-sig") as file:
            file.write(json.dumps(modelStruct, indent=4, ensure_ascii=False))


class JaniRModel(_BaseModel):
    def __init__(self, name, type, criterion, gamma, horizon):
        super().__init__(name, type)
        self._criterion = criterion
        self._gamma = gamma
        self._horizon = horizon

    @property
    def criterion(self):
        return self._criterion
    
    @property
    def gamma(self):
        return self._gamma
    
    @property
    def horizon(self):
        return self._horizon
    
    def _buildStateAndTransition(self):
        visitedStates = set()
        transitions = dict()
        deadlocks = set()
        maxSilentActionCounter = 0

        lowMemReprMap = dict()
        queue = deque()
        initStates = self.getInitStates()
        for initState in initStates:
            lowMemReprMap[initState] = initState.getLowMemRepr()
            queue.append(initState)
        
        edges = self._automata.edges
        locs = self._automata.locations
        funcGetter = self._functions
        while queue:
            state = queue.popleft()
            if state not in visitedStates:
                stateLowMemRepr = state.getLowMemRepr()
                visitedStates.add(state)


                print(len(visitedStates), end="\r")

                deadlock = True
                silentActionCounter = 0
                for edge in edges:
                    if not edge.isSatisfied(state, funcGetter):
                        continue

                    deadlock = False
                    source = edge.source
                    action = edge.action
                    if action == "silentAction":
                        action = f"{action}_{silentActionCounter}"
                        silentActionCounter += 1
                    for edgeDestination in edge.edgeDestinations:
                        nextState = state.clone(source,
                                                edgeDestination.destination,
                                                edgeDestination.assignments,
                                                locs,
                                                funcGetter)
                        probability = edgeDestination.probability.eval(varGetter = state,
                                                                       funcGetter = funcGetter)
                        reward = edgeDestination.reward.eval(varGetter = nextState,
                                                             funcGetter = funcGetter)
                        if probability <= 0.:
                            continue

                        nextStateLowMemRepr = lowMemReprMap.get(nextState)
                        if nextStateLowMemRepr is None:
                            lowMemReprMap[nextState] = nextState.getLowMemRepr()
                            nextStateLowMemRepr = lowMemReprMap[nextState]

                        transition = (stateLowMemRepr, nextStateLowMemRepr, action)
                        transitions[transition] = transitions.get(transition,
                                                                  np.array([0., 0.])) + [probability, reward]
                        if nextState not in visitedStates:
                            queue.append(nextState)
                        else:
                            del nextState
                if deadlock:
                    # deadlocks[(stateLowMemRepr, stateLowMemRepr, "deadlock")] = np.array([1., 0.])
                    deadlocks.add(stateLowMemRepr)
                maxSilentActionCounter = max(maxSilentActionCounter, silentActionCounter)
            else:
                del state
        actions = { f"silentAction_{i}" for i in range(maxSilentActionCounter) }
        actions.update(self._actions)

        return initStates, set(lowMemReprMap.values()), transitions, deadlocks, actions

    def _buildTransitionAndRewardMatrix(self, states, transitions, deadlocks, actions):
        transitionsPrime = {
            action: {
                state: dict() for state in states
            } for action in actions
        }
        for (state, statePrime, action), data in transitions.items():
            transitionsPrime[action][state][statePrime] = data

        stateMapIndex = { state: i for i, state in enumerate(states) }
        actionMapIndex  = { action: i for i, action in enumerate(actions) }

        penality = float("-inf") if self.criterion == "max" else float("inf")

        n, m = len(states), len(actions)
        stateSpace = mc.MarmoteInterval(0, n - 1)
        actionSpace = mc.MarmoteInterval(0, m - 1)

        Transitions = [ mc.SparseMatrix(n) for _ in range(m) ]
        Rewards = [ mc.SparseMatrix(n) for _ in range(m) ]
        for action in actions:
            actionIndex = actionMapIndex[action]
            for state in states:
                stateIndex = stateMapIndex[state]
                items = transitionsPrime[action][state].items()
                if items:
                    for statePrime, data in items:
                        statePrimeIndex = stateMapIndex[statePrime]

                        prob, rew = data

                        Transitions[actionIndex].addEntry(stateIndex, statePrimeIndex, prob)
                        Rewards[actionIndex].addEntry(stateIndex, statePrimeIndex, rew)
                else:
                    Transitions[actionIndex].addEntry(stateIndex, stateIndex, 1.)
                    if state not in deadlocks:
                        Rewards[actionIndex].addEntry(stateIndex, stateIndex, penality)
        return stateSpace, actionSpace, Transitions, Rewards

    def buildTransitionAndReward(self):
        initStates, states, transitions, deadlocks, actions = self._buildStateAndTransition()
        print(f"{len(states)} states")
        print(f"{len(actions)} actions")
        print(f"{len(transitions)} transitions and {len(deadlocks)} deadlocks")
        print(f"In total, {len(transitions) + len(deadlocks)} transitions")
        stateMapIndex = { state: i for i, state in enumerate(states) }
        print([stateMapIndex[s.getLowMemRepr()] for s in initStates])
        return self._buildTransitionAndRewardMatrix(states, transitions, deadlocks, actions)
