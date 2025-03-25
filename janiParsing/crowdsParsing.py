# import library
import json
from copy import deepcopy

import numpy as np
import marmote.core as mc
import marmote.markovchain as mmc
import networkx as nx


class Data:
    def __init__(self, jani_data):
        self.jani_data = jani_data
        self.variables = jani_data["variables"]
        self.varDict = {}
        for k, var in enumerate(self.variables):
            self.varDict[var['name']] = k
        self.constants = jani_data["constants"]
        self.constDict = {}
        for k, const in enumerate(self.constants):
            self.constDict[const['name']] = k

        # asseignment for two constants
        self.constants[2]["value"] = 3
        self.constants[3]["value"] = 5

    def getVariableType(self, variable):
        idx = self.varDict[variable]
        type = variables[idx]['type']
        if type == 'bool':
            return 'bool'
        else:
            return type["base"]

    def getVariableIdx(self, variable):
        return self.varDict[variable]

    def getConstantValue(self, const):
        idx = self.constDict[const]
        return self.constants[idx]['value']


def satisfyExpression(exp, node):
    if isinstance(exp, str):
        if data.getVariableType(exp) == 'bool':
            idx = data.getVariableIdx(exp)
            return node[idx]
    if exp['op'] == '∧':
        return satisfyExpression(exp['left'], node) and satisfyExpression(exp['right'], node)
    idx = data.getVariableIdx(exp['left'])
    if exp['op'] == '=':
        return calValue(node[idx]) == calValue(exp['right'])
    elif exp['op'] == '<':
        return calValue(node[idx]) < calValue(exp['right'])
    elif exp['op'] == '>':
        return calValue(node[idx]) > calValue(exp['right'])


def calValue(value):
    if isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return data.getConstantValue(value)
    else:
        if value["op"] == "+":

        elif value["op"] == "-":