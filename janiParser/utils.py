from janiParser.reader.reader import JaniReader
from janiParser.dataMarmote import DataMarmote

import pickle
import os

def buildAndSaveMDPModelFromQCompBenchmark(fullModelName: str, modelParams: dict={}, replace=False) -> list[str]:
    """Build and save one or more DataMarmote objects as `.pkl` files from a
    JANI MDP model available online at `https://qcomp.org/benchmarks/`.

    Each property defined in the model results in a separate `.pkl` file containing
    the corresponding DataMarmote object.

    Parameters:
        fullModelName (str): The full JANI model file name.

        modelParams (dict): Optional parameters used to configure a scalable model.

        replace (bool): If True, existing .pkl files will be overwritten.

    Returns:
        out (list[str]): The list of path to the saved .pkl files (one per property).
    """
    modelName = fullModelName.split(".")[0]
    URL = f"https://qcomp.org/benchmarks/mdp/{modelName}/{fullModelName}"
    model = JaniReader(URL, modelParams=modelParams, isLocalPath=False).build()

    rootPath = f"pickleFiles/{modelName}"
    if not os.path.exists(rootPath):
        os.makedirs(rootPath)

    dataMarmoteSavedPaths = []
    for prop in model.getPropertyNames():
        savedName = fullModelName.replace(".jani", f".{prop}.pkl")
        savedPath = f"{rootPath}/{savedName}"

        dataMarmoteSavedPaths.append(savedPath)
        if os.path.exists(savedPath) and not replace:
            continue

        MDPData = model.getMDPData(prop)
        dataMarmote = DataMarmote(MDPData)

        with open(savedPath, "wb") as file:
            pickle.dump(dataMarmote, file, pickle.HIGHEST_PROTOCOL)
        del MDPData
        del dataMarmote
    return dataMarmoteSavedPaths

def loadDataMarmoteFromPickleFile(path: str) -> DataMarmote:
    """Load a DataMarmote object from a `.pkl` file.
    
    Parameters:
        path (str): The path to the .pkl file.

    Returns
        out (DataMarmote): The deserialized DataMarmote object. 
    """
    with open(path, "rb") as file:
        dataMarmote = pickle.load(file)
    return dataMarmote

from time import time as _timer

import marmote.mdp as mmdp
import mdptoolbox.mdp

def _measureSolverMarmote(marmoteObject: mmdp.DiscountedMDP | mmdp.TotalRewardMDP | mmdp.AverageMDP,
                          resolMethodName: str, reps: int, _, epsilon, maxIter, maxInIter) -> float:
    methodMap = {
        "ValueIteration": lambda: marmoteObject.ValueIteration(epsilon, maxIter),
        "RelativeValueIteration": lambda: marmoteObject.RelativeValueIteration(epsilon, maxIter),
        "ValueIterationGS": lambda: marmoteObject.ValueIterationGS(epsilon, maxIter),
        "PolicyIterationModified": lambda: marmoteObject.PolicyIterationModified(epsilon, maxIter, epsilon, maxInIter),
        "PolicyIterationModifiedGS": lambda: marmoteObject.PolicyIterationModifiedGS(epsilon, maxIter, epsilon, maxInIter)
    }

    if resolMethodName not in methodMap:
        raise ValueError(f"'Marmote' library class {marmoteObject.className()} does not support resolution method {resolMethodName}")
    method = methodMap[resolMethodName]

    totalTime = 0.
    for _ in range(reps):
        startTime = _timer()
        method()
        totalTime += _timer() - startTime
    return totalTime / reps

def _measureSolverMDPToolbox(mdpData: tuple, resolMethodName: str, reps: int, discount, epsilon, maxIter, maxInIter):
    transitions, reward = mdpData
    methodMap = {
        "ValueIteration": lambda: mdptoolbox.mdp.ValueIteration(transitions, reward, discount, epsilon=epsilon, max_iter=maxIter),
        "RelativeValueIteration": lambda: mdptoolbox.mdp.RelativeValueIteration(transitions, reward, epsilon=epsilon, max_iter=maxIter),
        "ValueIterationGS": lambda: mdptoolbox.mdp.ValueIterationGS(transitions, reward, discount, epsilon=epsilon, max_iter=maxIter),
        "PolicyIteration": lambda: mdptoolbox.mdp.PolicyIteration(transitions, reward, discount, max_iter=maxIter),
        "PolicyIterationModified": lambda: mdptoolbox.mdp.PolicyIterationModified(transitions, reward, discount, epsilon=epsilon, max_iter=maxInIter)
    }

    if resolMethodName not in methodMap:
        raise ValueError(f"'MDPtoolbox' library does not support resolution method {resolMethodName}")
    methodConstructor = methodMap[resolMethodName]

    totalTime = 0.
    for _ in range(reps):
        method = methodConstructor()
        startTime = _timer()
        method.run()
        totalTime += _timer() - startTime
    return totalTime / reps

_supportedSolverMap = {
    "Marmote": _measureSolverMarmote,
    "MDPToolbox": _measureSolverMDPToolbox
}

def buildAndMeasureJaniMDPModel(fullModelName: str,
                                modelParams: dict={},
                                replace: bool=False,
                                solverName: str="Marmote",
                                resolMethods: list[str]=["ValueIteration", "PolicyIterationModified"],
                                discount: float=None,
                                epsilon: float=1e-3, maxIter: int=1000, maxInIter: int=10, reps: int=20) -> list[dict[str, str | int | float]]:
    """"""
    if solverName not in _supportedSolverMap:
        raise ValueError(f"Unsupported solver: {solverName}")
    runner = _supportedSolverMap[solverName]

    if discount is None:
        if solverName == "MDPToolbox":
            raise Exception("Solver 'MDPToolbox' requires a discount factor for resolution")
        mdpType = None
    elif 0 < discount < 1:
        mdpType = "DiscountedMDP"
    elif discount == 1:
        mdpType = "TotalRewardMDP"
    else:
        discount = 1
        mdpType = "AverageMDP"

    results = []
    savedPaths = buildAndSaveMDPModelFromQCompBenchmark(fullModelName, modelParams=modelParams, replace=replace)
    for path in savedPaths:
        modelName = os.path.splitext(path.split("/")[-1])[0]
        dataMarmote = loadDataMarmoteFromPickleFile(path)

        if solverName == "Marmote":
            if mdpType is not None:
                dataMarmote.setMDPTypeTo(mdpType, discount)
            createdObject = dataMarmote.createMarmoteObject()
        elif solverName == "MDPToolbox":
            createdObject = dataMarmote.buildTransitionRewardForMDPToolbox()
        
        data = { "name": modelName, "number-of-states": dataMarmote.nbStates(), "number-of-actions": dataMarmote.nbActions() }
        for methodName in resolMethods:
            try:
                data[methodName] = runner(createdObject, methodName, reps, discount, epsilon, maxIter, maxInIter)
            except Exception as e:
                print(e)
                data[methodName] = None
                continue
        results.append(data)
    return results
