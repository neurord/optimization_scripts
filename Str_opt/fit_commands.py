import ajustador as aju
from ajustador.helpers import converge
import os
def fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params,generations,popsiz, seed, test_size, map_func = None):
    last_slash = os.getcwd().rfind('/')
    tmpdir=os.getcwd()[0:last_slash]+'/fit'+modeltype+'-'+ntype+'-'+dirname

    fit = aju.optimize.Fit(tmpdir,
                        exp_to_fit,
                        modeltype, ntype,
                        fitness, params,
                        _make_simulation=aju.optimize.MooseSimulation.make,
                        _result_constructor=aju.optimize.MooseSimulationResult,
                        map_func = map_func)

    fit.load()
    fit.do_fit(generations, popsize=popsiz,seed=seed,sigma=2)
    return fit
