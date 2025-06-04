import os
import time
import io
import contextlib
import re
import sys
import threading
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

# roo_path = "../../benchmarks/prism2jani/"
#
# files = {
#    # "pacman.v2.jani": [{"MAXSTEPS": N} for N in range(5,46)],
#     "consensus.2.v1.jani": [{"K": N} for N in range(8, 417, 8)]
# }
#
# properties = {"pacman.v2.jani":["crash"], "consensus.2.v1.jani": ["c2", "steps_max", "discunt", "horizon"]}

roo_path = "./files/"

# files = {
#     "beb.3-4.v1.jani" : [{"N":3}],
#     "blocksworld.5.v1.jani" : [{}],
#     "cdrive.6.v1.jani" : [{}],
#     "cdrive.10.v1.jani" : [{}],
#     "csma.2-2.v1.jani" : [{}],
#     "csma.2-4.v1.jani" : [{}],
#     "elevators.a-3-3.v1.jani" : [{}],
#     "elevators.a-11-9.v1.jani" : [{}],
#     "elevators.b-3-3.v1.jani" : [{}],
#     "philosophers-mdp.3.v1.jani" : [{}],
#     "tireworld.17.v1.jani" : [{}],
# }
#
# properties = {
#     "beb.3-4.v1.jani" : ["LineSeized"],
#     "blocksworld.5.v1.jani" : ["goal"],
#     "cdrive.6.v1.jani" : ["goal"],
#     "cdrive.10.v1.jani" : ["goal"],
#     "csma.2-2.v1.jani" : ["all_before_max", "time_max"],
#     "csma.2-4.v1.jani" : ["all_before_max", "time_max"],
#     "elevators.a-3-3.v1.jani" : ["goal"],
#     "elevators.a-11-9.v1.jani" : ["goal"],
#     "elevators.b-3-3.v1.jani" : ["goal"],
#     "philosophers-mdp.3.v1.jani" : ["eat"],
#     "tireworld.17.v1.jani" : ["goal"]
# }

files = {
    "firewire_dl.v1.jani" : [{"delay" : 3, "deadline" : i} for i in range(200, 401, 10)],
    #"eajs.2.v1.jani" : [{"energy_capacity": i, "B":4} for i in range(100, 150, 5)]

    #有的instance 读取特别慢 eajs
    # 有的instance特别大，有的instance
}

properties = {
    "firewire_dl.v1.jani" : ["deadline", "discunt", "horizon"]
    #"eajs.2.v1.jani" : ["ExpUtil", "ProbUtil", "discunt", "horizon"]
}


def capture_output_c(fn, *args, **kwargs):
    old_stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(old_stdout_fd)

    r, w = os.pipe()
    os.dup2(w, old_stdout_fd)

    output = []

    def read_output():
        with os.fdopen(r) as reader:
            output.append(reader.read())

    t = threading.Thread(target=read_output)
    t.start()

    try:
        result = fn(*args, **kwargs)
    finally:
        os.close(w)
        os.dup2(saved_stdout_fd, old_stdout_fd)
        os.close(saved_stdout_fd)

    t.join()
    return result, output[0]

def function(fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        result, output = capture_output_c(fn, *args, **kwargs)
        total = round(time.perf_counter() - t0, 5)

        match = re.search(r"Done with (\d+) iterations and final distance=([\deE\.\+-]+)", output)
        if match:
            iters = int(match.group(1))
            dist = float(match.group(2))
        else:
            iters = None
            dist = None

        return total, result, iters, dist
    except Exception as e:
        return None, None, None, None


def run(file_name, param, props, rep=30):
    path = os.path.join(roo_path, file_name)

    try:
        model = rd.JaniReader(path, modelParams=param).build()
        for prop in props:
            try:
                if prop == "discunt":
                    MDPData = model.getMDPData(props[0])
                    dataMarmote = DataMarmote(MDPData)
                    mdp = dataMarmote.createMarmoteObject(discount=True)
                elif prop == "horizon":
                    MDPData = model.getMDPData(props[0])
                    dataMarmote = DataMarmote(MDPData)
                    mdp = dataMarmote.createMarmoteObject(horizonFini=True)
                else:
                    MDPData = model.getMDPData(prop)
                    dataMarmote = DataMarmote(MDPData)
                    mdp = dataMarmote.createMarmoteObject()
            except Exception as e:
                print(f"Error building MDP: {e}")
                continue

            mdp.changeVerbosity(True)
            n = len(MDPData["states"])
            a = len(MDPData["actions"])
            initIdx = dataMarmote.getInitStatesIdx()[0]
            mdp_type = mdp.className()
            is_average_mdp = "AverageMDP" in mdp_type
            is_finite = "FiniteHorizonMDP" in mdp_type

            for i in range(rep):
                result = {
                    "file": file_name, "params": str(param), "prop": str(prop),
                    "type": mdp_type, "n": n, "a": a,
                    "repeat": i + 1, "error": "",
                }

                try:
                    time_VI, opt1, it_VI, d_VI = function(mdp.ValueIteration, 1e-8, 5000)
                    result.update({
                        "time_VI": time_VI,
                        "iters_VI": it_VI,
                        "dist_VI": d_VI
                    })

                    if is_finite:
                        result.update({
                            "time_VIGS": None, "iters_VIGS": None, "dist_VIGS": None,
                            "time_RVI": None, "iters_RVI": None, "dist_RVI": None
                        })

                        for idx in range(1,4):
                            result[f"time_PI{idx}"] = None
                            result[f"iters_PI{idx}"] = None
                            result[f"dist_PI{idx}"] = None
                            result[f"time_PIGS{idx}"] = None
                            result[f"iters_PIGS{idx}"] = None
                            result[f"dist_PIGS{idx}"] = None
                        pd.DataFrame([result]).to_csv(output_file, mode="a", header=False, index=False)

                        continue

                    if is_average_mdp:
                        result.update({
                            "time_VIGS": None, "iters_VIGS": None, "dist_VIGS": None
                        })
                        time_RVI, opt5, it_RVI, d_RVI = function(mdp.RelativeValueIteration, 1e-8, 5000)
                        result.update({
                            "time_RVI": time_RVI,
                            "iters_RVI": it_RVI,
                            "dist_RVI": d_RVI
                        })

                        for idx in range(1, 4):
                            result[f"time_PI{idx}"] = None
                            result[f"iters_PI{idx}"] = None
                            result[f"dist_PI{idx}"] = None
                            result[f"time_PIGS{idx}"] = None
                            result[f"iters_PIGS{idx}"] = None
                            result[f"dist_PIGS{idx}"] = None
                        pd.DataFrame([result]).to_csv(output_file, mode="a", header=False, index=False)

                        continue
                    else:
                        time_VIGS, opt2, it_VIGS, d_VIGS = function(mdp.ValueIterationGS, 1e-8, 5000)
                        result.update({
                            "time_VIGS": time_VIGS,
                            "iters_VIGS": it_VIGS,
                            "dist_VIGS": d_VIGS
                        })
                        result.update({"time_RVI": None, "iters_RVI": None, "dist_RVI": None})

                    for idx, (eps, inner) in enumerate([(0.01, 500), (1e-4, 1500), (1e-7, 5000)], 1):
                        t_PI, opt3, it_PI, d_PI = function(mdp.PolicyIterationModified, 1e-8, 5000, eps, inner)
                        result[f"time_PI{idx}"] = t_PI
                        result[f"iters_PI{idx}"] = it_PI
                        result[f"dist_PI{idx}"] = d_PI

                        t_PIGS, opt4, it_PIGS, d_PIGS = function(mdp.PolicyIterationModifiedGS, 1e-8, 5000, eps, inner)
                        result[f"time_PIGS{idx}"] = t_PIGS
                        result[f"iters_PIGS{idx}"] = it_PIGS
                        result[f"dist_PIGS{idx}"] = d_PIGS

                    # result[f"time_PI{3}"] = None
                    # result[f"iters_PI{3}"] = None
                    # result[f"dist_PI{3}"] = None
                    # result[f"time_PIGS{3}"] = None
                    # result[f"iters_PIGS{3}"] = None
                    # result[f"dist_PIGS{3}"] = None

                except Exception as e:
                    result["error"] = str(e)

                pd.DataFrame([result]).to_csv(output_file, mode="a", header=False, index=False)

    except Exception as e:
        print(e)



if __name__ == '__main__':
    output_file = ("firewire_test.csv")
    completed = set()

    header = [
        "file", "params", "prop", "type", "n", "a",
        "repeat", "error",
        "time_VI", "iters_VI", "dist_VI",
        "time_VIGS", "iters_VIGS", "dist_VIGS",
        "time_RVI", "iters_RVI", "dist_RVI",
    ]

    for i in [1, 2, 3]:
        header += [
            f"time_PI{i}", f"iters_PI{i}", f"dist_PI{i}",
            f"time_PIGS{i}", f"iters_PIGS{i}", f"dist_PIGS{i}"
        ]

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(",".join(header) + "\n")

    for file_name, param_list in files.items():
        for param in param_list:
            key = (file_name, str(param))
            run(file_name, param, properties[file_name], rep=20)
            print(f"finished: {file_name} param: {param}")