from variable import Type, Constant, Variable
from expression import Expression
from automaton import EdgeDestination, Edge, Automaton
from model import Model

class JANIRSyntaxError(SyntaxError):
    pass

import json
from collections import deque

UNARY_EXPRESSION = {
    '¬',
    'floor',
    'ceil',
    'abs',
    'sgn',
    'trc'
}

BINARY_EXPRESSION = {
    '∨',
    '∧',
    '⇒',
    '=',
    '≠',
    '<',
    '≤',
    '>',
    '≥',
    '+',
    '-',
    '*',
    '%',
    '/',
    'pow',
    'log',
    'min',
    'max'
}

import marmote.core as mc
import marmote.mdp as mmdp

def replaceVariableName(model: Model, name: str, scope):
    # If 'name' is declared as a global variable then return 'name',
    # otherwise return 'name.automaName'
    if model.isGlobal(name):
        return name
    _, automaName = scope
    return f"{name}.{automaName}"

import numpy as np

class Reader(object):
    def __init__(self, path, modelParameters):
        self.path = path
        self.modelParameters = modelParameters

    ###########################################################################
    ###########################################################################
    def build(self):
        with open(self.path, "r", encoding="utf-8-sig") as file:
            parsedStruct = json.loads(file.read())
        model = self.parseModel(parsedStruct)
        print("Parser OK")
        # return model
        # print(model.nonTransientVars)
        states, actions, transitions = self.buildStateSpace(model)
        
        # print("Build state Ok")
        # self.buildMarmoteInstance(model, states, actions, transitions)
        print(f"{len(states)} states, {len(actions)} actions, {len(transitions)} transitions")

    def buildMarmoteInstance(self, model:Model, states, actions, transitions):
        # Just a test

        # transitions = { (s, s', a): [p, r] }

        # map action or state -> index
        stateToIndex = { state: i for i, state in enumerate(states) }
        actionToIndex = { action: i for i, action in enumerate(actions) }
        
        # n: state size, m: action size
        n, m = len(states), len(actions)
        
        stateSpace = mc.MarmoteInterval(0, n - 1)
        actionSpace = mc.MarmoteInterval(0, m - 1)

        Transitions = [ mc.SparseMatrix(n) for _ in range(m) ]
        Reward = mc.FullMatrix(n, m)
        

        for (state, nextState, action), array in transitions.items():
            prob, reward = array
            # print(prob, reward)

            actionIndex = actionToIndex[action]
            sIndex = stateToIndex[state]
            sPrimeIndex = stateToIndex[nextState]

            Reward.addToEntry(sIndex, actionIndex, reward)

            Transitions[actionIndex].addEntry(sIndex, sPrimeIndex, prob)

        mdp = mmdp.AverageMDP("max", stateSpace, actionSpace, Transitions, Reward)
        opt = mdp.ValueIteration(1e-6, 500)
        with open("out.txt", "w") as file:
            print(opt, file=file)
        print(f"initial State = {stateToIndex[model.getInitState()]}")
        pass

    def buildStateSpace(self, model: Model):
        visitedStates = set()
        transitions = dict()
        maxSilentActionCnt = 0

        queue = deque()
        queue.append(model.getInitState())

        automaton = model.getSyncAutoma()
        while queue:
            state = queue.popleft()
            if state not in visitedStates:
                visitedStates.add(state)

                deadlock = True
                silentActionCounter = 0
                for edge in automaton.edges:
                    if not edge.isSatisfied(state):
                        continue
                    deadlock = False

                    source = edge.source
                    action = edge.action
                    if action == "silentAction":
                        action = f"{action}_{silentActionCounter}"
                        silentActionCounter += 1
                    # print("#####################")
                    for edgeDestination in edge.edgeDestinations:
                        nextState = state.clone(source,
                                                edgeDestination.destination,
                                                edgeDestination.assignments)
                        
                        prob = edgeDestination.probability.eval(state)

                        reward = edgeDestination.reward.eval(state)
                        # print(edgeDestination.assignments)
                        # print(nextState)
                        transition = (state, nextState, action)
                        transitions[transition] = transitions.get(transition, np.array([0., 0.], dtype=np.float64)) + np.array([prob, prob * reward], dtype=np.float64)
                        # if transition not in transitions:
                        #     transitions[transition] = prob
                        # else:
                        #     transitions[transition] += prob

                        if nextState not in visitedStates:
                            queue.append(nextState)
                if deadlock:
                    # pass
                    transitions[(state, state, "deadlock")] = np.array([1., 0.], dtype=np.float64)
                
                maxSilentActionCnt = max(maxSilentActionCnt, silentActionCounter)
        

        actions = { f"silentAction_{i}" for i in range(maxSilentActionCnt) }
        actions.update(model.getActions())
        actions.add("deadlock")

        # print(model.getActions())
        # print(actions, maxSilentActionCnt, model.getActions())
        return visitedStates, actions, transitions

    def parseModel(self, parsedStruct: dict):
        if "name" not in parsedStruct:
            raise JANIRSyntaxError("A model must have a name")
        name = parsedStruct["name"]

        model = Model(name, "mdp")

        for actionStruct in parsedStruct.get("actions", []):
            if "name" not in actionStruct:
                raise JANIRSyntaxError("An action must hava a name")
            model.addAction(actionStruct["name"])
        
        if "system" not in parsedStruct:
            raise JANIRSyntaxError(f"a system section is required by model '{name}'")
        self.parseSystem(model, parsedStruct["system"])

        self.parseConstants(model, parsedStruct.get("constants", []))
        self.parseVariables(model, parsedStruct.get("variables", []), ("global", ))
        
        if "automata" not in parsedStruct:
            raise JANIRSyntaxError(f"a automata section is required by model '{name}'")
        for automaStruct in parsedStruct["automata"]:
            self.parseAutoma(model, automaStruct)
        model.synchronize()
        return model
    
    ###########################################################################
    ###########################################################################
    def parseSystem(self, model: Model, systemStruct):
        automaToIndex = dict()
        syncActionsList = list()

        if "elements" not in systemStruct:
            raise SyntaxError()
        
        for i, elemStruct in enumerate(systemStruct["elements"]):
            if "automaton" not in elemStruct:
                raise SyntaxError()
            automaToIndex[elemStruct["automaton"]] = i
        
        if "syncs" in systemStruct:
            for i, syncStruct in enumerate(systemStruct["syncs"]):
                result = syncStruct.get("result", f"silentAction_{i}.sync")

                syncActions = list()
                for action in syncStruct.get("synchronise", []):
                    if action is not None and not model.containsAction(action):
                        raise Exception(f"Unrecognized action '{action}' by model '{model.name}'")
                    syncActions.append(action)
                syncActionsList.append((result, syncActions))
        model.setSystemInformations(automaToIndex, syncActionsList)
        # print(automaToIndex, syncActionsList)

    # to do
    def parseExpression(self, model: Model, exprStruct, scope) -> Expression:
        if isinstance(exprStruct, int):
            return Expression("int", exprStruct)
        elif isinstance(exprStruct, float):
            return Expression("real", exprStruct)
        elif isinstance(exprStruct, bool):
            return Expression("bool", exprStruct)
        elif isinstance(exprStruct, str):
            if model.isConstant(exprStruct):
                return self.parseExpression(model, model.getConstantValue(exprStruct), scope)
            return Expression("var", replaceVariableName(model, exprStruct, scope))
        else:
            if "op" not in exprStruct:
                raise SyntaxError()
            op = exprStruct["op"]
            if op in UNARY_EXPRESSION:
                if "exp" not in exprStruct:
                    raise SyntaxError()
                exp = self.parseExpression(model, exprStruct["exp"], scope)
                return Expression(op, exp)
            elif op in BINARY_EXPRESSION:
                if "left" not in exprStruct or "right" not in exprStruct:
                    raise SyntaxError()
                left = self.parseExpression(model, exprStruct["left"], scope)
                right = self.parseExpression(model, exprStruct["right"], scope)
                return Expression.reduceExpr(op, left, right)
            elif op == "ite":
                pass
            elif op == "call":
                pass
            else:
                raise Exception()

    def parseType(self, model: Model, typeStruct, scope) -> Type:
        if isinstance(typeStruct, str):
            if typeStruct not in  [ "int", "real", "bool" ]:
                raise Exception()
            return Type(typeStruct)
        
        if "kind" not in typeStruct:
            raise SyntaxError()
        kind = typeStruct["kind"]
        if kind != "bounded":
            raise Exception()
        
        if "base" not in typeStruct:
            raise SyntaxError()
        base = typeStruct["base"]
        if base not in [ "int", "real" ]:
            raise Exception()
        
        if "lower-bound" not in typeStruct or "upper-bound" not in typeStruct:
            raise SyntaxError()
        lowBound = self.parseExpression(model, typeStruct["lower-bound"], scope)
        upBound = self.parseExpression(model, typeStruct["upper-bound"], scope)

        if not lowBound.isConstExpr() or not upBound.isConstExpr():
            raise SyntaxError()
        lowBound = lowBound.eval(None)
        upBound = upBound.eval(None)
        return Type(base, boundaries=(lowBound, upBound))

    def parseConstants(self, model: Model, constsStruct):
        scope = ("global", )
        for constStruct in constsStruct:
            if "name" not in constStruct:
                raise SyntaxError("A constant must have a name")
            name = constStruct["name"]

            if "type" not in constStruct:
                raise SyntaxError(f"Constant '{name}' must have a type")
            type = self.parseType(model, constStruct["type"], scope)

            if "value" not in constStruct:
                if name not in self.modelParameters:
                    raise Exception()
                value = self.modelParameters[name]
            else:
                value = self.parseExpression(model, constStruct["value"], scope)
                if not value.isConstExpr():
                    raise Exception()
                value = value.eval(None)
            
            model.addConstant(Constant(name, type, value))
    
    def parseVariables(self, model: Model, varsStruct, scope):
        isGlobalDecl = scope[0] == "global"

        for varStruct in varsStruct:
            name = varStruct["name"]
            if not isGlobalDecl:
                name = f"{name}.{scope[1]}"

            type = self.parseType(model, varStruct["type"], scope)

            transient = varStruct.get("transient", False)
            
            if "initial-value" not in varStruct:
                raise SyntaxError()
            else:
                initValue = self.parseExpression(model, varStruct["initial-value"], scope)
                if not initValue.isConstExpr():
                    raise Exception()
                initValue = initValue.eval(None)
 
            model.addVariable(Variable(name, type, scope, initValue, transient))

    def parseAutoma(self, model: Model, automatonStruct):
        name = automatonStruct["name"]

        scope = ("local", name)

        if "variables" in automatonStruct:
            self.parseVariables(model, automatonStruct["variables"], scope)

        locations = self.parseLocations(model, automatonStruct["locations"], scope)
        initLocations = set(map(lambda loc: f"{loc}.{name}", automatonStruct.get("initial-locations", {})))

        edges = []
        for edgeStruct in automatonStruct["edges"]:
            edges.append(self.parseEdge(model, edgeStruct, scope))
        model.addAutomaton(Automaton(name, locations, initLocations, edges))

    def parseLocations(self, model, locationsStruct, scope):
        _, automataName = scope

        locations = dict()
        for locStruct in locationsStruct:
            if "name" not in locStruct:
                raise JANIRSyntaxError()
            name = locStruct["name"]
            name = f'{name}.{automataName}'

            transientValues = dict()
            for transientValuesStruct in locStruct.get("transient-values", []):
                if "ref" not in transientValuesStruct:
                    raise JANIRSyntaxError()
                ref = transientValuesStruct["ref"]

                # must no ref to transient variable
                if "value" not in transientValuesStruct:
                    raise JANIRSyntaxError()
                expr = self.parseExpression(model, transientValuesStruct["value"], scope)
                transientValues[ref] = expr
            locations[name] = transientValues
        return locations
        
    def parseEdge(self, model: Model, edgeStruct, scope):
        _, automataName = scope

        if "location" not in edgeStruct:
            raise JANIRSyntaxError()
        source = edgeStruct["location"]
        source = { f"{source}.{automataName}" }
        
        action = edgeStruct.get("action", "silentAction")

        if "guard" not in edgeStruct:
            guard = Expression("bool", True)
        else:
            guardStruct = edgeStruct["guard"]
            if "exp" not in guardStruct:
                raise JANIRSyntaxError()
            guard = self.parseExpression(model, guardStruct["exp"], scope)
        
        if "destinations" not in edgeStruct:
            raise JANIRSyntaxError()
        edgeDestinations = []
        for destStruct in edgeStruct["destinations"]:
            if "location" not in destStruct:
                raise JANIRSyntaxError()
            dest = destStruct["location"]
            dest = { f"{dest}.{automataName}" }

            if "probability" not in destStruct:
                prob = Expression("real", 1.)
            else:
                probStruct = destStruct["probability"]
                if "exp" not in probStruct:
                    raise JANIRSyntaxError()
                # print(self.parseExpression(model, probStruct["exp"], scope))
                prob = self.parseExpression(model, probStruct["exp"], scope)


            if "reward" not in destStruct:
                reward = Expression("int", 0)
            else:
                rewardStruct = destStruct["reward"]
                reward = self.parseExpression(model, rewardStruct["exp"], scope)


            assignments = dict()
            for assignStruct in destStruct.get("assignments", []):
                if "ref" not in assignStruct:
                    raise JANIRSyntaxError()
                ref = assignStruct["ref"]
                if not model.isGlobal(ref):
                    ref = f"{ref}.{automataName}"

                if "value" not in assignStruct:
                    raise JANIRSyntaxError()
                expr = self.parseExpression(model, assignStruct["value"], scope)
                assignments[ref] = expr
            edgeDestinations.append(EdgeDestination(dest, prob, assignments, reward))
        return Edge(source, action, guard, edgeDestinations)


if __name__ == "__main__":
    path = "../../benchmarks/modest2jani/beb.3-4.v1.jani"
    reader = Reader(path, {"N":3})
    reader.build()
