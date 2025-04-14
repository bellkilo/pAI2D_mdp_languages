import json

from model import JaniModel, JaniRModel
from variable import Type, Constant, Variable
from expression import Expression
from automata import Automata, Edge, EdgeDestination
from function import Function
from property import Property

# from collections import deque
import numpy as np

import marmote.core as mc
import marmote.mdp as mmdp

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

class _BaseReader(object):
    def __init__(self, path, modelParams):
        assert isinstance(modelParams, dict)
        self._path = path
        self._modelParams = modelParams
    
    def build(self):
        """Build the model."""
        with open(self._path, "r", encoding="utf-8-sig") as file:
            modelStruct = json.loads(file.read())
        return self.parseModel(modelStruct)
    
    def parseExpression(self, model, expression, scope, funcParams=set()):
        # Parse a constant expression.
        if isinstance(expression, int):
            return Expression("int", expression)
        if isinstance(expression, float):
            return Expression("real", expression)
        if isinstance(expression, bool):
            return Expression("bool", expression)
        # Parse a variable expression.
        if isinstance(expression, str):
            # Replace a constant variable with its associated value.
            if model.isConstantVariable(expression):
                return self.parseExpression(model, model.getConstantValue(expression), scope)
            
            # Parse global variable, local variable and function parameters.
            if model.isGlobalVariable(expression) or expression in funcParams:
                name = expression
            elif len(scope) == 2:
                _, autamata = scope
                name = f"{expression}.{autamata}"
                if not model.isLocalVariable(name):
                    raise SyntaxError(f"Unrecognized variable name '{expression}' for model '{model.name}'")
            else:
                raise SyntaxError(f"Unrecognized variable name '{expression}' for model '{model.name}'")
            return Expression("var", name)
        # Parse complex expression.
        if "op" not in expression:
            raise SyntaxError("A complex expression must have an operator")
        operator = expression["op"]
        # Parse unary expression.
        if operator in UNARY_EXPRESSION:
            if "exp" not in expression:
                raise SyntaxError("An unary expression must have an exp structure")
            return Expression.reduceExpression(operator,
                                               self.parseExpression(model, expression["exp"], scope, funcParams))
        # Parse binary expression.
        if operator in BINARY_EXPRESSION:
            if "left" not in expression:
                raise SyntaxError("A binary expression must have a left structure")
            if "right" not in expression:
                raise SyntaxError("A binary expression must have a right structure")
            return Expression.reduceExpression(operator,
                                                    self.parseExpression(model, expression["left"], scope, funcParams),
                                                    self.parseExpression(model, expression["right"], scope, funcParams))
        # Parse if-else expression.
        if operator == "ite":
            if "if" not in expression:
                raise SyntaxError("An if-else expression must have an if structure")
            if "then" not in expression:
                raise SyntaxError("An if-else expression must have a then structure")
            if "else" not in expression:
                raise SyntaxError("An if-else expression must have an else structure")
            return Expression.reduceExpression("ite",
                                               self.parseExpression(model, expression["if"], scope, funcParams),
                                               self.parseExpression(model, expression["then"], scope, funcParams),
                                               self.parseExpression(model, expression["else"], scope, funcParams))
        # Parse function call expression.
        if operator == "call":
            if "function" not in expression:
                raise SyntaxError("A function call expression must have a function identifier")
            name = expression["function"]

            if not model.containsFunction(name):
                raise SyntaxError(f"Unrecognized function name '{name}' for model '{model.name}'")
            if "args" not in expression:
                raise SyntaxError(f"Function '{name}' must have an args structure, even if the function takes no arguments")
            args = [
                self.parseExpression(model, arg, scope, funcParams)
                for arg in expression["args"]
            ]
            # Check that the number of arguments corresponds to the function definition.
            if len(model.getFunction(name).parameters) != len(args):
                raise SyntaxError(f"Unmatched argument size for function '{name}'")
            return Expression("call", name, args)
        raise SyntaxError(f"Unrecognized operator '{operator}' for model '{model.name}'")
    
    def parseType(self, model, type, scope):
        # Parse basic type definition.
        if isinstance(type, str):
            if type not in ["int", "real", "bool"]:
                raise Exception(f"Unsupported type '{type}'")
            return Type(type)
        # Parse complex (bounded) type definition.
        if "kind" not in type:
            raise SyntaxError("A complex type must have a kind")
        kind = type["kind"]
        if kind != "bounded":
            raise Exception(f"Unsupported kind '{kind}' for model '{model.name}'")
        
        if "base" not in type:
            raise SyntaxError("A complex type must have a base")
        base = type["base"]
        if base not in ["int", "real"]:
            raise Exception(f"Unsupproted base '{base}' for model '{model.name}'")
        
        if "lower-bound" not in type:
            raise SyntaxError("A complex type (bounded) must have a lower bound")
        if "upper-bound" not in type:
            raise SyntaxError("A complex type (bounded) must have a upper bound")
        lowerBound = self.parseExpression(model, type["lower-bound"], scope)
        upperBound = self.parseExpression(model, type["upper-bound"], scope)
        if not lowerBound.isConstExpression():
            raise Exception("Lower bound must be a constant expression.")
        if not upperBound.isConstExpression():
            raise Exception("Upper bound must be a constant expression.")
        lowerBound = lowerBound.eval()
        upperBound = upperBound.eval()
        return Type(base, boundaries=(lowerBound, upperBound))
    
    def parseConstants(self, model, constants):
        scope = ("global",)
        for constant in constants:
            if "name" not in constant:
                raise SyntaxError("A constant variable must have a name")
            name = constant["name"]

            if "type" not in constant:
                raise SyntaxError(f"Constant variable '{name}' must have a type structure")
            type = self.parseType(model, constant["type"], scope)

            if "value" not in constant:
                # If the value expression is not given, then it must be given as the model parameters.
                if name not in self._modelParams:
                    raise Exception(f"Missing constant value '{name}' for model '{model.name}'")
                value = self._modelParams[name]
            else:
                value = self.parseExpression(model, constant["value"], scope)
                if not value.isConstExpression():
                    raise Exception(f"Constant value must ba a constant expression for variable '{name}'")
                value = value.eval()
            model.addConstant(Constant(name, type, value))
    
    def parseVariables(self, model, variables, scope):
        isGlobalScope = scope[0] == "global"
        for variable in variables:
            if "name" not in variable:
                raise SyntaxError("A variable must have a name")
            name = variable["name"]
            if not isGlobalScope:
                name = f"{name}.{scope[1]}"
            
            if "type" not in variable:
                raise SyntaxError(f"Variable '{name}' must have a type structure")
            type = self.parseType(model, variable["type"], scope)

            transient = variable.get("transient", False)

            if "initial-value" not in variable:
                if not transient and (type.boundaries is None or type.type == "real"):
                    raise SyntaxError(f"Variable without initial value must be bounded")
                initValue = None
            else:
                initValue = self.parseExpression(model, variable["initial-value"], scope)
                if not initValue.isConstExpression():
                    raise Exception(f"Initial value must be a constant expression for variable '{name}'")
                initValue = initValue.eval()
            model.addVariable(Variable(name, type, scope, initValue, transient))
    
    def getVarRefsFromExprStruct(self, model, targetExpression):
        """Return all variable references in an expression structure."""
        varRefs = set()
        stack = [targetExpression]
        while stack:
            expression = stack.pop()
            if isinstance(expression, str) and not model.isConstantVariable(expression):
                varRefs.add(expression)
            elif isinstance(expression, dict):
                operator = expression["op"]
                if operator in UNARY_EXPRESSION:
                    stack.append(expression["exp"])
                elif operator in BINARY_EXPRESSION:
                    stack.append(expression["left"])
                    stack.append(expression["right"])
                elif operator == "ite":
                    stack.append(expression["if"])
                    stack.append(expression["then"])
                    stack.append(expression["else"])
                elif operator == "call":
                    stack.append(expression["function"])
                    stack.extend(expression["args"])
        return varRefs

    def parseLocations(self, model, locations, scope):
        _, autamata = scope
        locs = dict()
        for location in locations:
            if "name" not in location:
                raise SyntaxError("A location must have a name")
            name = location["name"]
            name = f"{name}.{autamata}"

            transientValues = dict()
            for transientValue in location.get("transient-values", []):
                if "ref" not in transientValue:
                    raise SyntaxError("A transient value structure must have a reference")
                ref = transientValue["ref"]
                if model.isLocalVariable(ref):
                    ref = f"{ref}.{autamata}"
                if not model.isTransientVariable(ref):
                    raise SyntaxError(f"Non-transient variable reference '{ref}'")
                
                if "value" not in transientValue:
                    raise SyntaxError("A transient value structure must have a value expression")
                value = transientValue["value"]

                # The value expression for a transient value structure must not refer to another transient variable.
                varRefs = self.getVarRefsFromExprStruct(model, value)
                if varRefs and np.all(list(map(model.isTransientVariable, varRefs))):
                    raise SyntaxError(f"A value expression for a transient value must not refer to another transient variable")
                value = self.parseExpression(model, value, scope)
                transientValues[ref] = value
            locs[name] = transientValues
        return locs

    def parseAutomata(self, model, automata):
        if "name" not in automata:
            raise SyntaxError("An automata must have a name")
        name = automata["name"]
        scope = ("local", name)

        self.parseVariables(model, automata.get("variables", []), scope)
        self.parseFunctions(model, automata.get("functions", []), scope)

        if "locations" not in automata:
            raise SyntaxError(f"Automata '{name}' must have a locations structure")
        locs = self.parseLocations(model, automata["locations"], scope)

        if "initial-locations" not in automata:
            raise SyntaxError(f"Automata '{name}' must have a initial-locations structure")
        initLocs = automata["initial-locations"]
        if len(initLocs) == 0:
            raise SyntaxError(f"Automata '{name}' must have at least 1 initial location")
        if len(initLocs) > 1:
            raise SyntaxError(f"Model '{model.name}' does not support multiple initial location")
        initLoc = { f"{initLocs[0]}.{name}"}

        if "edges" not in automata:
            raise SyntaxError(f"Automata '{name}' must have an edges structure")
        edges = [ self.parseEdge(model, edge, scope) for edge in automata["edges"] ]
        model.addAutomata(Automata(name, locs, initLoc, edges))

    def parseEdge(self, model, edge, scope):
        _, automata = scope
        if "location" not in edge:
            raise SyntaxError(f"An edge in automata '{automata}' must have a location")
        src = edge["location"]
        src = { f"{src}.{automata}" }

        action = edge.get("action", "silentAction")

        if "guard" not in edge:
            guard = Expression("bool", True)
        else:
            guard = edge["guard"]
            if "exp" not in guard:
                raise SyntaxError(f"A guard structure in automata '{automata}' must have a exp structure")
            guard = self.parseExpression(model, guard["exp"], scope)
        
        if "destinations" not in edge:
            raise SyntaxError(f"An edge in automata '{automata}' must have a destinations structure")
        edgeDests = list()
        for destination in edge["destinations"]:
            if "location" not in destination:
                raise SyntaxError(f"An edge destination in automata '{automata}' must have a location")
            dest = destination["location"]
            dest = { f"{dest}.{automata}" }

            if "probability" not in destination:
                prob = Expression("real", 1.)
            else:
                prob = destination["probability"]
                if "exp" not in prob:
                    raise SyntaxError(f"A probability structure in automata '{automata}' must have a exp structure")
                prob = self.parseExpression(model, prob["exp"], scope)
            
            if "reward" not in destination:
                reward = Expression("int", 0)
            else:
                reward = destination["reward"]
                if "exp" not in reward:
                    raise SyntaxError(f"A reward structure in automata '{automata}' must have a exp structure")
                reward = self.parseExpression(model, reward["exp"], scope)
            
            assgns = dict()
            for assgn in destination.get("assignments", []):
                if "ref" not in assgn:
                    raise SyntaxError(f"An assignment structure in automata '{automata}' must have a reference")
                ref = assgn["ref"]
                if model.isLocalVariable(ref):
                    ref = f"{ref}.{automata}"

                if "value" not in assgn:
                    raise SyntaxError(f"An assignment structure in automata '{automata}' must have a value expression")
                value = self.parseExpression(model, assgn["value"], scope)
                assgns[ref] = value
            edgeDests.append(EdgeDestination(dest, prob, assgns, reward))
        return Edge(src, action, guard, edgeDests)

    def parseSystem(self, model, system):
        automataMapIndex = dict()
        syncActionsList = list()
        if "elements" not in system:
            raise SyntaxError("System must have an elements structure")
        for i, element in enumerate(system["elements"]):
            if "automaton" not in element:
                raise SyntaxError("A element structure must have an automaton structure")
            automataMapIndex[element["automaton"]] = i
        
        if "syncs" in system:
            for i, sync in enumerate(system["syncs"]):
                result = sync.get("result", f"silentAction_{i}.sync")
                syncActions = []
                for action in sync.get("synchronise", []):
                    if action is not None and not model.containsAction(action):
                        raise SyntaxError(f"Unrecognized action '{action}' for model '{model.name}'")
                    syncActions.append(action)
                syncActionsList.append((result, syncActions))
        model.setSystemInformation(automataMapIndex, syncActionsList)

    def parseFunctions(self, model, functions, scope):
        isGlobalScope = scope[0] == "global"
        # First parse that adds all function declarations.
        for function in functions:
            if "name" not in function:
                raise SyntaxError("A function must have a name")
            name = function["name"]
            if not isGlobalScope:
                assert len(scope) == 2
                name = f"{name}.{scope[1]}"
            model.declareFunction(name)
        # Second parse that adds the function definitions.
        funcDependencies = dict()
        for function in functions:
            name = function["name"]
            if not isGlobalScope:
                name = f"{name}.{scope[1]}"
            
            if "type" not in function:
                raise SyntaxError(f"Function '{name}' must have a type structure")
            type = self.parseType(model, function["type"], scope)

            if "parameters" not in function:
                raise SyntaxError(f"Function '{name}' must have a parameters structure, "
                                  "even if function takes no arguments")
            # Parse function parameters.
            params = dict()
            for parameter in function["parameters"]:
                if "name" not in parameter:
                    raise SyntaxError(f"Function parameter must have a name")
                paramName = parameter["name"]
                if paramName in params:
                    raise SyntaxError(f"Duplicate parameter '{paramName}' in function '{name}'")
                if model.containsVariable(paramName):
                    raise SyntaxError(f"Parameter name '{paramName}' conflicts with existing variable")
                
                if "type" not in parameter:
                    raise SyntaxError(f"Function parameter '{paramName}' must have a type structure")
                params[paramName] = self.parseType(model, parameter["type"], scope)
            # Parse function body expression.
            
            if "body" not in function:
                raise SyntaxError(f"Function '{name}' must have a body structure")
            body = self.parseExpression(model, function["body"], scope, params)

            # add function dependencies.
            varRefs = self.getVarRefsFromExprStruct(model, function["body"])
            funcDependencies[name] = set(filter(model.containsFunction, varRefs))
            model.addFunction(Function(name, type, scope, params, body))
        # Check function dependencies.
        self.checkFunctionDependencies(funcDependencies)

    def checkFunctionDependencies(self, funcDependencies):
        """Check function dependencies."""
        def hasCircularDependency(v, status):
            if status[v] is not None:
                return not status[v]
            status[v] = False
            for u in funcDependencies[v]:
                if hasCircularDependency(u, status):
                    return True
            status[v] = True
            return False
        status = { v: None for v in funcDependencies }
        for v in funcDependencies:
            if status[v] is None and hasCircularDependency(v, status):
                raise Exception(f"Detect a circular dependency in function '{v}'")

    def parseModel(self, modelStruct):
        pass


class JaniReader(_BaseReader):
    def __init__(self, path, modelParams):
        super().__init__(path, modelParams)

    def parseModel(self, modelStruct):
        if "name" not in modelStruct:
            raise SyntaxError("A model must have a name")
        name = modelStruct["name"]

        if "type" not in modelStruct:
            raise SyntaxError(f"Model '{name}' must have a type")
        type = modelStruct["type"]
        if type not in ["mdp", "dtmc"]:
            raise Exception(f"Unsupported type '{type}' for model '{name}'")
        
        model = JaniModel(name, type)

        for action in modelStruct.get("actions", []):
            if "name" not in action:
                raise SyntaxError("An action must have a name")
            model.addAction(action["name"])
        
        if "system" not in modelStruct:
            raise SyntaxError(f"Model '{name}' must have a system structure")
        self.parseSystem(model, modelStruct["system"])

        self.parseConstants(model, modelStruct.get("constants", []))
        self.parseVariables(model, modelStruct.get("variables", []), ("global", ))
        self.parseFunctions(model, modelStruct.get("functions", []), ("global", ))

        if "automata" not in modelStruct:
            raise SyntaxError(f"Model '{name}' must have a automata structure")
        for automata in modelStruct["automata"]:
            self.parseAutomata(model, automata)
        
        # TODO
        self.parseProperties(model, modelStruct["properties"])
        return model
    # TODO
    # hard code
    def parseProperties(self, model: JaniModel, properties):
        scope = ("global", )
        for property in properties:
            if "name" not in property:
                raise SyntaxError("A property must have a name")
            name = property["name"]

            if "expression" not in property:
                raise SyntaxError(f"Property '{name}' must have a property expression")
            propertyExpression = property["expression"]

            if "op" not in propertyExpression:
                raise SyntaxError("A property expression must have an operator")
            operator = propertyExpression["op"]
            if operator != "filter":
                raise SyntaxError("The top level operator must be 'filter'")
            
            if "values" not in propertyExpression:
                raise SyntaxError()
            values = propertyExpression["values"]

            operator = values["op"]
            if operator == "Pmax" or operator == "Pmin":
                criterion = operator[1:]
                if values["exp"]["op"] == "F":
                    reward = self.parseExpression(model, values["exp"]["exp"], scope)
                elif values["exp"]["op"] == "U":
                    if values["exp"]["left"]:
                        reward = self.parseExpression(model, values["exp"]["right"], scope)
                    else:
                        left = self.parseExpression(model, values["exp"]["left"], scope)
                        right = self.parseExpression(model, values["exp"]["right"], scope)
                        # ...
                        reward = Expression(
                                    "ite",
                                    left,
                                    Expression("int", 0),
                                    Expression(
                                        "ite",
                                        right,
                                        Expression("int", 1),
                                        Expression("int", -1)
                                    )
                                )
                else:
                    raise Exception(f"Unimplemented")
            elif operator == "Emax" or operator == "Emax":
                pass
            else:
                pass
            model.addProperty(Property(name, criterion, reward))
            
        pass


class JaniRReader(_BaseReader):
    def __init__(self, path, modelParameters):
        super().__init__(path, modelParameters)
    
    def parseModel(self, modelStruct):
        if "name" not in modelStruct:
            raise SyntaxError("A model must have a name")
        name = modelStruct["name"]

        if "criterion" not in modelStruct:
            raise SyntaxError(f"Model '{name}' must have a criterion")
        criterion = modelStruct["criterion"]
        if criterion not in ["max", "min"]:
            raise SyntaxError(f"Unsupported criterion '{criterion}'")
        
        if "type" not in modelStruct:
            raise SyntaxError(f"Model '{name}' must have a type")
        type = modelStruct["type"]
        gamma = 1
        horizon = float("inf")
        if type == "DiscountedMDP":
            if "discounted-factor" not in modelStruct:
                raise SyntaxError("A discounted MDP must have a discounted factor")
            gamma = modelStruct["discounted-factor"]
            if gamma < 0 or gamma >= 1:
                raise SyntaxError("A discounted factor must be in interval [0, 1)")
        elif type == "AverageMDP":
            pass
        elif type == "TotalRewardMDP":
            pass
        elif type == "FiniteHorizonMDP":
            if "horizon" not in modelStruct:
                raise SyntaxError("A finite horizon MDP must have a horizon")
            horizon = modelStruct["horizon"]
        else:
            raise Exception("Unimplemented yet!!")
        model = JaniRModel(name, type, criterion, gamma, horizon)

        for action in modelStruct.get("actions", []):
            if "name" not in action:
                raise SyntaxError("An action must have a name")
            model.addAction(action["name"])
        
        if "system" not in modelStruct:
            raise SyntaxError(f"Model '{name}' must have a system structure")
        self.parseSystem(model, modelStruct["system"])

        self.parseConstants(model, modelStruct.get("constants", []))
        self.parseVariables(model, modelStruct.get("variables", []), ("global", ))
        self.parseFunctions(model, modelStruct.get("functions", []), ("global", ))

        if "automata" not in modelStruct:
            raise SyntaxError(f"Model '{name}' must have a automata structure")
        for automata in modelStruct["automata"]:
            self.parseAutomata(model, automata)

        model.synchronize()
        return model


if __name__ == "__main__":
    ###########################################################
    # PPDDL instances

    # 462400 states
    # 22 actions
    # 3851520 transitions and 0 deadlocks
    # In total, 3851520 transitions
    # python3 reader.py  1149,72s user 3,50s system 99% cpu 19:17,56 total

    # path = "../../benchmarks/ppddl2jani/zenotravel.4-2-2.v1.jani"
    # reader = JaniReader(path, modelParams={})

    ###########################################################
    # Prism instances

    # 1296 states
    # 3 actions
    # 2412 transitions and 0 deadlocks
    # In total, 2412 transitions
    # python3 reader.py  0,44s user 0,02s system 99% cpu 0,470 total
    # path = "../../benchmarks/prism2jani/consensus.2.v1.jani"
    # reader = JaniReader(path, modelParams={ "K": 10 })

    path = "../../benchmarks/prism2jani/csma.2-2.v1.jani"
    reader = JaniReader(path, modelParams={})

    # 12828 states
    # 1 actions
    # 21795 transitions and 0 deadlocks
    # In total, 21795 transitions
    # python3 reader.py  343,71s user 0,28s system 99% cpu 5:44,92 total
    # path = "../../benchmarks/prism2jani/eajs.2.v1.jani"
    # reader = JaniReader(path, modelParams={ "energy_capacity": 100, "B": 100 })

    # 956 states
    # 6 actions
    # 3696 transitions and 0 deadlocks
    # In total, 3696 transitions
    # python3 reader.py  0,50s user 0,02s system 99% cpu 0,522 total
    # path = "../../benchmarks/prism2jani/philosophers-mdp.3.v1.jani"
    # reader = JaniReader(path, modelParams={})


    # 96894 states
    # 10 actions
    # 129170 transitions and 0 deadlocks
    # In total, 129170 transitions
    # python3 reader.py  1714,69s user 2,12s system 99% cpu 28:43,84 total
    # path = "../../benchmarks/prism2jani/pacman.v2.jani"
    # reader = JaniReader(path, modelParams={ "MAXSTEPS": 15 })

    reader.build().writeJaniR("out.txt", "all_before_max")
    model = JaniRReader("out.txt", {}).build()
    stateSpace, actionSpace, Transitions, Rewards = model.buildTransitionAndReward()
    mdp = mmdp.TotalRewardMDP(model.criterion, stateSpace, actionSpace, Transitions, Rewards)
    with open("out_1.txt", "w") as file:
        print(mdp.ValueIteration(1e-10, 1000), file=file)
