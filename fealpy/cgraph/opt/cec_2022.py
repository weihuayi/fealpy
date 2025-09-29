from ..nodetype import CNodeType, PortConf, DataType

class CEC2022(CNodeType):
    TITLE: str = "CEC2022"
    PATH: str = "优化.cec_2022"
    INPUT_SLOTS = [
        PortConf("func_num", DataType.INT),
        PortConf("dim", DataType.MENU, default=2, items=[2, 10, 20]),
        PortConf("opt_alg", DataType.MENU, default="GeneticAlg", items=[
            "GeneticAlg", 
            "ParticleSwarmOpt", 
            "CrayfishOptAlg", 
            "HoneybadgerAlg", 
            "SnowAblationOpt", 
            "RimeOptAlg"
        ]),
        PortConf("NP", DataType.INT),
        PortConf("MaxIT", DataType.INT, default=1000)
    ]
    OUTPUT_SLOTS = [
        PortConf("Optimal_sol", DataType.TENSOR),
        PortConf("Optimal_val", DataType.FLOAT)
    ]

    @staticmethod
    def run(func_num, dim, opt_alg, NP, MaxIT):
        from fealpy import opt
        from fealpy.opt import initialize, opt_alg_options
        from fealpy.opt.model import OPTModelManager

        options = {
            'func_num': func_num,
            'dim': dim
        }
        manager = OPTModelManager('single')
        problem = manager.get_example(3, **options)
        lb, ub = problem.get_bounds()
        x0 = initialize(NP, dim, ub, lb)

        options = opt_alg_options(
            x0,
            lambda x: problem.evaluate(x),
            (lb, ub),
            NP,
            MaxIters=MaxIT
        )

        OptClass = getattr(opt, opt_alg)
        optimizer = OptClass(options)

        optimizer.run()
        
        return optimizer.gbest, optimizer.gbest_f 

