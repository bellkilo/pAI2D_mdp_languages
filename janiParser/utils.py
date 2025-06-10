from janiParser.reader.reader import JaniReader
from janiParser.dataMarmote.dataMarmote import DataMarmote

from typing import List

import pickle
import os

def buildAndSaveMDPModelFromQcompBenchmarks(fullModelName: str, modelParams: dict, replace=False) -> List[str]:
    modelName = fullModelName.split(".")[0]
    URL = f"https://qcomp.org/benchmarks/mdp/{modelName}/{fullModelName}"
    model = JaniReader(URL, modelParams=modelParams, isLocalPath=False).build()

    rootPath = f"pickleFiles/{modelName}"
    if not os.path.exists(rootPath):
        os.makedirs(rootPath)

    dataMarmoteSavedPaths = []
    for prop in model.getPropertyNames():
        savedName = fullModelName.replace("jani", f"{prop}.pkl")
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
    with open(path, "rb") as file:
        dataMarmote = pickle.load(file)
    return dataMarmote
