import os
import time
import pandas as pd

import src.reader.reader as rd
from src.dataMarmote.dataMarmote import DataMarmote
import marmote.core as marmotecore



# export PYTHONPATH=/mnt/c/Users/PC/Music/pAI2D_mdp_languages/

roo_path = "../../benchmarks/mcJani/"

files = {
    "brp.jani" : [{"N": N, "MAX": 20}
                 for N in range(1, 1001, 10)]
    # "brp.jani": [{"N" : 16, "MAX": 2},
    #              {"N" : 16, "MAX": 3},
    #              {"N" : 16, "MAX": 4},
    #              {"N" : 16, "MAX": 5},
    #              {"N" : 32, "MAX": 2},
    #              {"N" : 32, "MAX": 3},
    #              {"N" : 32, "MAX": 4},
    #              {"N" : 32, "MAX": 5},
    #              {"N" : 64, "MAX": 2},
    #              {"N" : 64, "MAX": 3},
    #              {"N" : 64, "MAX": 4},
    #              {"N" : 64, "MAX": 5}],
    # "egl.jani": [{"N" : 5, "L": 2},
    #              {"N" : 5, "L": 4},
    #              {"N" : 5, "L": 6},
    #              {"N" : 5, "L": 8}],
    # # "haddad-monmege.jani": [{"N" : 20, "p": 0.7},
    # #                         {"N" : 100, "p": 0.7},   taill trop petite
    # #                         {"N" : 300, "p": 0.7}],
    # # "herman.3.jani": [{}],
    # # "herman.5.jani": [{}],
    # # "herman.7.jani": [{}],
    # "herman.9.jani": [{}],
    # "herman.11.jani": [{}],
    # "herman.13.jani": [{}],
    # # "herman.15.jani": [{}],
    # # "herman.17.jani": [{}],
    # # "herman.19.jani": [{}],
    # # "herman.21.jani": [{}],
    # # "leader_sync.3-2.jani": [{}],
    # # "leader_sync.3-3.jani": [{}],
    # # "leader_sync.3-4.jani": [{}],
    # # "leader_sync.4-2.jani": [{}],
    # "leader_sync.4-3.jani": [{}],
    # "leader_sync.4-4.jani": [{}],
    # "leader_sync.5-2.jani": [{}],
    # "leader_sync.5-3.jani": [{}],
    # "leader_sync.5-4.jani": [{}],
    # "nand.jani": [{"N": 20, "K": 1},
    #               {"N": 20, "K": 2},
    #               {"N": 20, "K": 3},
    #               {"N": 20, "K": 4},
    #               {"N": 40, "K": 1},
    #               {"N": 40, "K": 2},
    #               {"N": 40, "K": 3},
    #               {"N": 40, "K": 4},
    #               {"N": 60, "K": 1}
    #               ],
    # "oscillators.3-6-0.1-1.jani": [{"mu": 0.1, "lambda":1}],
    # "oscillators.6-6-0.1-1.jani": [{"mu": 0.1, "lambda": 1}],
    # "oscillators.6-8-0.1-1.jani": [{"mu": 0.1, "lambda": 1}],
    # "oscillators.6-10-0.1-1.jani": [{"mu": 0.1, "lambda": 1}],
    # "oscillators.7-10-0.1-1.jani": [{"mu": 0.1, "lambda": 1}],
    # "oscillators.8-8-0.1-1.jani": [{"mu": 0.1, "lambda": 1}],
    # "oscillators.8-10-0.1-1.jani": [{"mu": 0.1, "lambda": 1}]
}


def function(fn, *args, **kwargs):
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    total = time.perf_counter() - t0
    return round(total, 5)

def run(file_name, param, rep = 30):
    path = os.path.join(roo_path, file_name)
    results = []

    try:
        t_read_start = time.perf_counter()
        model = rd.JaniReader(path, modelParams=param).build()
        mcData = model.getMCData()

        n = len(mcData["states"])
        m = mcData["number-transitions"]
        dm = DataMarmote(mcData)
        mc = dm.createMarmoteObject()
        t_read_end = time.perf_counter()
        time_read = round(t_read_end - t_read_start, 5)


        for i in range(rep):
            time_SD   = function(mc.StationaryDistribution)
            time_RLGL = function(
                mc.StationaryDistributionRLGL,
                1000, 1e-10,
                marmotecore.UniformDiscreteDistribution(0, n - 1),
                False
            )

            result = {
                "file": file_name,
                "params": str(param),
                "n": n,
                "m": m,
                "m/n": round(m/n, 4) if n else None,
                "time_read": time_read,
                "time_SD": time_SD,
                "time_RLGL": time_RLGL,
                "error": "",
                "repeat" : i+1
            }
            results.append(result)
    except Exception as e:
        print(f"error: {str(e)}")
        result = {
            "file": file_name,
            "params": str(param),
            "n": None,
            "m": None,
            "m/n": None,
            "time_SD": None,
            "time_RLGL": None,
            "error": str(e),
            "repeat": None
        }
        results.append(result)

    return results


if __name__ == '__main__':
    # output_file = "../../src/mcBenchMarks/brp_test_results.csv"
    output_file = "brp_test_results.csv"
    completed = set()
    if os.path.exists(output_file):
        df_done = pd.read_csv(output_file)
        completed = set(zip(df_done["file"], df_done["params"]))


    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("file,params,n,m,m/n,time_read,time_SD,time_RLGL,error,repeat\n")

    for file_name, param_list in files.items():
        for param in param_list:
            key = (file_name, str(param))
            if key in completed:
                print(f"skip: {file_name}, param: {param}")
                continue

            results = run(file_name, param)
            df = pd.DataFrame(results)
            df.to_csv(output_file, mode="a", header=False, index=False)
            print(f"finished: {file_name} param: {param}")

# file,params,n,m,m/n,time_SD,time_RLGL,error
# brp.jani,"{'N': 16, 'MAX': 2}",677,867,1.2806,0.4017,0.0003,0.0051,
# brp.jani,"{'N': 16, 'MAX': 3}",886,1155,1.3036,0.5469,0.0004,0.0062,
# brp.jani,"{'N': 16, 'MAX': 4}",1095,1443,1.3178,0.6653,0.0005,0.0075,
# brp.jani,"{'N': 16, 'MAX': 5}",1304,1731,1.3275,0.8153,0.0006,0.0087,
# brp.jani,"{'N': 32, 'MAX': 2}",1349,1731,1.2832,0.8342,0.0011,0.0198,
# brp.jani,"{'N': 32, 'MAX': 3}",1766,2307,1.3063,1.0545,0.0014,0.022,
# brp.jani,"{'N': 32, 'MAX': 4}",2183,2883,1.3207,1.3874,0.0019,0.0258,
# brp.jani,"{'N': 32, 'MAX': 5}",2600,3459,1.3304,1.6392,0.0024,0.0298,
# brp.jani,"{'N': 64, 'MAX': 2}",2693,3459,1.2844,1.6078,0.0044,0.0817,
# brp.jani,"{'N': 64, 'MAX': 3}",3526,4611,1.3077,2.1749,0.0058,0.0852,
# brp.jani,"{'N': 64, 'MAX': 4}",4359,5763,1.3221,2.6861,0.0073,0.0958,
# brp.jani,"{'N': 64, 'MAX': 5}",5192,6915,1.3319,3.4107,0.0096,0.1083,
# egl.jani,"{'N': 5, 'L': 2}",33790,34813,1.0303,385.18,0.0147,0.5814,
# egl.jani,"{'N': 5, 'L': 4}",74750,75773,1.0137,846.625,0.1117,0.0205,
# egl.jani,"{'N': 5, 'L': 6}",115710,116733,1.0088,1303.8894,0.4195,1.9858,
# egl.jani,"{'N': 5, 'L': 8}",156670,157693,1.0065,1761.2122,1.0348,2.6916,
# herman.9.jani,{},512,19684,38.4453,13.0779,0.0022,0.0038,
# herman.11.jani,{},2048,177148,86.498,209.7984,0.022,0.0217,
# herman.13.jani,{},8192,1594324,194.6196,3514.3114,0.5982,0.1488,
