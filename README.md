# optimization_scripts
Repository for python scripts used for parameter optimizations
In all cases, the optimization code is ajustador and the model is in moose_nerp
1. gp_opt has scripts for optimizating globus pallidus models, uses:
   * data: waves/gpedata-experimental, described in gpedata_experimental.py
   * moose_nerp package is moose_nerp/gp
2. Str_opt has scripts for optimizating spiny projection neuron models, uses:
   * data: waves/measurements1 or waves/A2Acredata described in measurements1.py and A2Acre.py, respectively
   * moose_nerp package is moose_nerp/d1d2
   * also will optimize fsi neurons using moose_nerp/fsi
3. EP_opt has scripts for optimizating entopeduncular neuron models, uses:
   * data: waves/EPmeasurements
   * moose_nerp package is moose_nerp/ep described in EPdata.py

other scripts include
1. chanvar.py. 
   * read in results of a set of optimizations and calculate mean and stdev of the parameters.  
   * Useful for initial assessment of whether some parameter values are cell class specific, or finding variance to use in network models
2. plot_opt_results.py
   * plot fit history and traces of best fit from a set of optimizations
   * optionally, create new moose_nerp package with the new parameters
3. generate_opt_script.py
   * automates creation of sets of optimization scripts
   * each script differs by neuron type, target data, or random seed
4. randomcalssifier.py
   * reads in parameters for a set of optimizations
   * uses random forest cluster analysis to determine which parameter values predict neuron type
5. analysis.py
   * reads in parameters for a set of optimizations
   * evaluates correlations among parameters
   * optionally creates output file of best parameters and fit histories from entire set for graphing