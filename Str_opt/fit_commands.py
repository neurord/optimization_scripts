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
    if test_size>0:
        mean_dict,std_dict,CV=converge.iterate_fit(fit,test_size,popsiz,std_crit=0.01,slope_crit=1e-3,max_evals=100000)
        return fit,mean_dict,std_dict,CV
    else:
        return fit,[],[],[]
