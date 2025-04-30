import os
import time
import pandas as pd

import src.reader.reader as rd
from src.dataMarmote.dataMarmote import DataMarmote
import marmote.core as marmotecore

# roo_path = "../../benchmarks/prism2jani/"
#
# files = {
#     "consensus.2.v1.jani": [{"K": N} for N in range(8, 721, 8)]
# }

# properties = ["steps_max"]

roo_path = "../../benchmarks/prism2jani/"

files = {
    "pacman.v2.jani": [{"MAXSTEPS": N} for N in range(5,46)]
}

properties = ["crash"]

def function(fn, *args, **kwargs):
    t0 = time.perf_counter()
    opt = fn(*args, **kwargs)
    total = time.perf_counter() - t0
    return round(total, 5), opt

def run(file_name, param, properties, rep=30):
    path = os.path.join(roo_path, file_name)
    results = []

    try:
        model = rd.JaniReader(path, modelParams=param).build()
        for prop in properties:
            t_parse_start = time.perf_counter()
            MDPData = model.getMDPData(prop)
            n = len(MDPData["states"])
            a = len(MDPData["actions"])
            dataMarmote = DataMarmote(MDPData)
            mdp = dataMarmote.createMarmoteObject()
            t_parse = time.perf_counter() - t_parse_start

            initIdx = dataMarmote.getInitStatesIdx()[0]
            mdp_type = mdp.className()
            is_average_mdp = 'AverageMDP' in mdp_type

            for i in range(rep):
                try:
                    time_VI, opt1 = function(mdp.ValueIteration, 1e-10, 2000)
                    if is_average_mdp:
                        time_VIGS = None
                        opt2 = None
                        time_RVI, opt5 = function(mdp.RelativeValueIteration, 1e-10, 2000)
                    else:
                        time_VIGS, opt2 = function(mdp.ValueIterationGS, 1e-10, 2000)
                        time_RVI = None
                        opt5 = None

                    time_PI, opt3 = function(mdp.PolicyIterationModified, 1e-10, 2000, 0.001, 2000)
                    time_PIGS, opt4 = function(mdp.PolicyIterationModifiedGS, 1e-10, 2000, 0.001, 2000)
                    error = ""
                except Exception as algo_error:
                    time_VI = time_VIGS = time_PI = time_PIGS = time_RVI = None
                    opt1 = opt2 = opt3 = opt4 = opt5 = None
                    error = str(algo_error)

                result = {
                    "file": file_name,
                    "params": str(param),
                    "prop": str(prop),
                    "type": mdp_type,
                    "n": n,
                    "a": a,
                    "time_read": t_parse,
                    "time_VI": time_VI,
                    "opt_VI": opt1.getValueIndex(initIdx) if opt1 else None,
                    "time_VIGS": time_VIGS,
                    "opt_VIGS": opt2.getValueIndex(initIdx) if opt2 else None,
                    "time_PI": time_PI,
                    "opt_PI": opt3.getValueIndex(initIdx) if opt3 else None,
                    "time_PIGS": time_PIGS,
                    "opt_PIGS": opt4.getValueIndex(initIdx) if opt4 else None,
                    "time_RVI": time_RVI,
                    "opt_RVI": opt5.getValueIndex(initIdx) if opt5 else None,
                    "repeat": i + 1,
                    "error": error
                }
                results.append(result)

    except Exception as e:
        print(e)
        result = {
            "file": file_name,
            "params": str(param),
            "prop": "",
            "type": "",
            "n": None,
            "a": None,
            "time_read": None,
            "time_VI": None,
            "opt_VI": None,
            "time_VIGS": None,
            "opt_VIGS": None,
            "time_PI": None,
            "opt_PI": None,
            "time_PIGS": None,
            "opt_PIGS": None,
            "time_RVI": None,
            "opt_RVI": None,
            "repeat": 0,
            "error": str(e)
        }
        results.append(result)

    return results



if __name__ == '__main__':
    output_file = "pacman_test_results_crash.csv"
    completed = set()

    if os.path.exists(output_file):
        df_done = pd.read_csv(output_file)
        completed = set(zip(df_done["file"], df_done["params"]))
    else:
        with open(output_file, "w") as f:
            f.write(",".join([
                "file", "params", "prop", "type", "n", "a", "time_read",
                "time_VI", "opt_VI", "time_VIGS", "opt_VIGS", "time_PI", "opt_PI",
                "time_PIGS", "opt_PIGS","time_RVI","opt_RVI",
                "repeat", "error"
            ]) + "\n")

    for file_name, param_list in files.items():
        for param in param_list:
            key = (file_name, str(param))
            if key in completed:
                print(f"skip: {file_name}, param: {param}")
                continue

            results = run(file_name, param, properties, rep=30)
            df = pd.DataFrame(results)
            df.to_csv(output_file, mode="a", header=False, index=False)
            print(f"finished: {file_name} param: {param}")