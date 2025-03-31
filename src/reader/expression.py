import numpy as np
from typing import Union

class Expression(object):
    __slots__ = ("__kind", "__args", "__arity")

    __OPERATORS = {
        "∨": np.logical_or,
        "∧": np.logical_and,
        "¬": np.logical_not,

        "⇒": lambda x, y: np.logical_or(np.logical_not(x), y),

        "=": np.equal,
        "≠": np.not_equal,
        "<": np.less,
        "≤": np.less_equal,
        ">": np.greater,
        "≥": np.greater_equal,

        "+": np.add,
        "-": np.subtract,
        "*": np.multiply,
        "%": np.mod,
        "/": np.divide,
        "pow": np.power,
        "log": np.log,
        "floor": np.floor,
        "ceil": np.ceil,
        "abs": np.abs,
        "sgn": np.sign,
        "min": np.minimum,
        "max": np.maximum,
        "trc": np.trunc
    }
    
    def __init__(self,
                 kind: str,
                 *args: Union['Expression', int, float, bool]):
        self.__kind = kind
        self.__args = args
        self.__arity = len(self.__args)

    def eval(self, struct):
        kind = self.__kind
        if kind in ("int", "real", "bool"):
            return self.__args[0]
        
        elif kind == "var":
            if self.__args[0] == "line_seized":
                print(struct.get(self.__args[0]))
            return struct.get(self.__args[0])
        
        elif kind == "ite":
            condition, ifExpr, elseExpr = self.__args
            if condition.eval(struct):
                return ifExpr.eval(struct)
            return elseExpr.eval(struct)

        elif kind == "call":
            pass

        elif kind in self.__OPERATORS:
            operator = self.__OPERATORS[kind]
            if self.__arity == 1:
                return operator(self.__args[0].eval(struct))
            else:
                left, right = self.__args
                return operator(left.eval(struct), right.eval(struct))
        else:
            raise Exception()

    def isConstExpr(self):
        return self.__kind in ("int", "real", "bool")
    
    @staticmethod
    def createConstExpr(value):
        if isinstance(value, (int, np.integer)):
            return Expression("int", value)
        if isinstance(value, (float, np.floating)):
            return Expression("real", value)
        if isinstance(value, (bool, np.bool_)):
            return Expression("bool", value)
        raise Exception(f"Unrecognized value type'{type(value)}'")
    
    @staticmethod
    def reduceExpr(op, *expr: 'Expression'):
        arity = len(expr)
        operator = Expression.__OPERATORS.get(op)
        # if arity == 1:
        #     # to do
        #     pass
        if arity == 2:
            left, right = expr
            if left.isConstExpr() and right.isConstExpr():
                value = operator(left.eval(None), right.eval(None))
                return Expression.createConstExpr(value)
            return Expression(op, left, right)
        # else:
        #     # to do
        #     pass


    @staticmethod
    def mergeExpression(expr1: 'Expression', expr2: 'Expression'):
        return Expression.reduceExpr("∧", expr1, expr2)
    
    def __str__(self):
        match self.__arity:
            case 1:
                if self.__kind in [ "int", "real", "bool", "var" ]:
                    return str(self.__args[0])
                return f"{self.__kind}{self.__args[0]}"
            case 2:
                left, right = self.__args
                return f"({left}) {self.__kind} ({right})"
            case _:
                pass
        return
    
    def __repr__(self):
        return self.__str__()
    
if __name__ == "__main__":
    from itertools import product
    expr = Expression("⇒",
                           Expression("∧",
                                           Expression("∨",
                                                           Expression("var", "a"),
                                                           Expression("var", "c")),
                                           Expression("∨",
                                                           Expression("var", "b"),
                                                           Expression("var", "c"))),
                           Expression("⇒",
                                           Expression("¬", Expression("var", "b")),
                                           Expression("∨",
                                                           Expression("∧",
                                                                           Expression("var", "a"),
                                                                           Expression("var", "b")),
                                                           Expression("var", "c"))))
    
    print(np.all([expr.eval({ "a": a, "b": b, "c": c }) for a, b, c in product([True, False], repeat=3)]))
