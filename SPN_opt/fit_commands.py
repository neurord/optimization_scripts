import ajustador as aju
from ajustador.helpers import converge

def fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params,generations,popsiz, seed, test_size):
    tmpdir='/tmp/fit'+modeltype+'-'+ntype+'-'+dirname
    
    fit = aju.optimize.Fit(tmpdir,
                        exp_to_fit,
                        modeltype, ntype,
                        fitness, params,
                        _make_simulation=aju.optimize.MooseSimulation.make,
                        _result_constructor=aju.optimize.MooseSimulationResult)

    fit.load()
    fit.do_fit(generations, popsize=popsiz,seed=seed)
    if test_size>0:
        mean_dict,std_dict,CV=converge.iterate_fit(fit,test_size,popsiz)
        return fit,mean_dict,std_dict,CV
    else:
        return fit,[],[],[]
