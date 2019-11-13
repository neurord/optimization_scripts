import ajustador as aju

def fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params,generations,popsiz, seed):
    tmpdir='/tmp/fit'+modeltype+'-'+ntype+'-'+dirname
    tmpdir=dirname
    fit = aju.optimize.Fit(tmpdir,
                           exp_to_fit,
                           modeltype, ntype,
                           fitness, params,
                           _make_simulation=aju.optimize.MooseSimulation.make,
                           _result_constructor=aju.optimize.MooseSimulationResult)
    
    fit.load()
    fit.do_fit(generations, popsize=popsiz,seed=seed)
    return fit
