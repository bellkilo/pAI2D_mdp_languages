from src.reader.reader import *
from src.dataMarmote.dataMarmote import *

if __name__ == "__main__":
    ###########################################################
    # path = "./benchmarks/modest2jani/beb.3-4.v1.jani"
    # model = JaniReader(path, modelParams={ "N": 1 }).build()
    # # MDPData = model.getMDPData("LineSeized")
    # MDPData = model.getMDPData("GaveUp")
    ###########################################################

    ###########################################################
    # path = "./benchmarks/ppddl2jani/blocksworld.5.v1.jani"
    # path = "./benchmarks/ppddl2jani/cdrive.2.v1.jani"
    # path = "./benchmarks/ppddl2jani/elevators.a-3-3.v1.jani"
    # path = "./benchmarks/ppddl2jani/exploding-blocksworld.5.v1.jani"
    # path = "./benchmarks/ppddl2jani/rectangle-tireworld.5.v1.jani"
    # path = "./benchmarks/ppddl2jani/tireworld.17.v1.jani"
    # path = "./benchmarks/ppddl2jani/triangle-tireworld.9.v1.jani"
    # path = "./benchmarks/ppddl2jani/zenotravel.4-2-2.v1.jani"
    # model = JaniReader(path).build()
    # MDPData = model.getMDPData("goal")
    ###########################################################

    ###########################################################
    # path = "./benchmarks/prism2jani/consensus.2.v1.jani"
    # model = JaniReader(path, modelParams={ "K": 10 }).build()
    # MDPData = model.getMDPData("steps_max")

    # path = "./benchmarks/prism2jani/csma.2-2.v1.jani"
    # # # path = "./benchmarks/prism2jani/csma.2-4.v1.jani"
    # model = JaniReader(path).build()
    # # MDPData = model.getMDPData("all_before_max")
    # # MDPData = model.getMDPData("all_before_min")
    # MDPData = model.getMDPData("some_before")

    path = "./benchmarks/prism2jani/philosophers-mdp.3.v1.jani"
    model = JaniReader(path).build()
    MDPData = model.getMDPData("eat")

    # path = "./benchmarks/prism2jani/zeroconf.v1.jani"
    # model = JaniReader(path, modelParams={ "reset": False, "N": 1, "K": 1 }).build()
    # MDPData = model.getMDPData("correct_max")
    ###########################################################

    # dataMarmote = DataMarmote(MDPData)
    # mdp = dataMarmote.createMarmoteObject()
    # opt = mdp.ValueIteration(1e-10, 2000)
    # for idx in dataMarmote.getInitStatesIdx():
    #     print(opt.getValueIndex(idx))

    # dataMarmote.saveAsJaniRFile("test.janir")

    # TODO
    # Really long, need to optimize.
    # dataMarmote.saveAsJaniRFile("test.janir")
    # model = JaniRReader("test.janir").build()
    # dataMarmote = DataMarmote(model.getMDPData())
    # mdp = dataMarmote.createMarmoteObject()
    # opt = mdp.ValueIteration(1e-6, 1000)
    # for idx in dataMarmote.getInitStatesIdx():
    #     print(opt.getValueIndex(idx))


    dim_SS = 4
    dim_AS = 3
    s =  mc.MarmoteBox([2, 2])
    a =  mc.MarmoteInterval(0,2)
    P0 = mc.SparseMatrix(dim_SS)

    P0.setEntry(0,1,0.875)
    P0.setEntry(0,2,0.0625)
    P0.setEntry(0,3,0.0625)
    P0.setEntry(1,1,0.75)
    P0.setEntry(1,2,0.125)
    P0.setEntry(1,3,0.125)
    P0.setEntry(2,2,0.5)
    P0.setEntry(2,3,0.5)
    P0.setEntry(3,3,1.0)
    print(P0.getEntry(0, 3))

    P1 =  mc.SparseMatrix(dim_SS)
    P1.setEntry(0,1,0.875)
    P1.setEntry(0,2,0.0625)
    P1.setEntry(0,3,0.0625)
    P1.setEntry(1,1,0.75)
    P1.setEntry(1,2,0.125)
    P1.setEntry(1,3,0.125)
    P1.setEntry(2,1,1.0)
    P1.setEntry(3,3,1.0)

    P2 =  mc.SparseMatrix(dim_SS)
    P2.setEntry(0,1,0.875)
    P2.setEntry(0,2,0.0625)
    P2.setEntry(0,3,0.0625)
    P2.setEntry(1,0,1.0)
    P2.setEntry(2,0,1.0)
    P2.setEntry(3,0,1.0)

    tms = [P0, P1, P2]

    rms = mc.FullMatrix(dim_SS, dim_AS)
    rms.setEntry(0,0,0)
    rms.setEntry(0,1,4000)
    rms.setEntry(0,2,6000)
    rms.setEntry(1,0,1000)
    rms.setEntry(1,1,4000)
    rms.setEntry(1,2,6000)
    rms.setEntry(2,0,3000)
    rms.setEntry(2,1,4000)
    rms.setEntry(2,2,6000)
    rms.setEntry(3,0,3000)
    rms.setEntry(3,1,4000)
    rms.setEntry(3,2,6000)

    c="min"
    mdp1 =  mmdp.AverageMDP(c, s, a, tms,rms)
    print(mdp1)

    dataMarmote = DataMarmote.fromMarmoteMDP(c, "AverageMDP", s, a, tms, rms)
    mdp = dataMarmote.createMarmoteObject()
    print(mdp)

    dataMarmote.saveAsJaniRFile("test.janir")

    model = JaniRReader("test.janir").build()
    dataMarmote2 = DataMarmote(model.getMDPData())
    mdp2 = dataMarmote2.createMarmoteObject()
    print(mdp2)

    for idx in range(dim_SS):
        print(s.DecodeState(idx), dataMarmote.idxToState(idx), dataMarmote2.idxToState(idx))
    pass