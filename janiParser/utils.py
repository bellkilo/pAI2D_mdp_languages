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
