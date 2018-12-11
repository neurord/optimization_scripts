import ajustador as aju
from ajustador.helpers import save_params,converge
from ajustador import drawing
import gpedata_experimental as gpe
import os
#must be in current working directory for this import to work, else use exec
import params_fitness_chan,fit_commands

########### Optimization of GP neurons ##############3
modeltype='gp'
rootdir='/home/avrama/moose/gp_opt/'
#use 1 and 3 for testing, 200 and 8 for optimization
generations=200
popsiz=8
seed=938
#after generations, do 25 more at a time and test for convergence
test_size=25

################## neuron /data specific specifications #############
ntype='arky'
morph_file='GP_arky_41comp.p'
#dataname is an experiment from data in gpedata_experimental; 0,2,4 specify a subset of traces
dataname='arky140'
exp_to_fit = gpe.data[dataname+'-2s'][[0,2,4]]

#subdirectory for putting output about fit
dirname='chan_'+dataname+str(seed)
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
os.chdir(rootdir+dirname)

######## set up parameters and fitness to be used for all opts  ############
params1,fitness=params_fitness_chan.params_fitness(morph_file,ntype,modeltype)

# set-up and do the optimization 
fit1,mean_dict1,std_dict1,CV1=fit_commands.fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params1,generations,popsiz, seed, test_size)

###########look at results
drawing.plot_history(fit1, fit1.measurement)

#Save parameters of good results toward the end, and all fitness values
startgood=0  #set to 0 to print/save all
threshold=10  #set to high value to print/save all

save_params.save_params(fit1, startgood, threshold)
#save_params.persist(fit1,'.')

'''
Repeat arky120, arky140, proto079,proto122 optimizations with channel kinetics allowed to vary 
but with limitations to axonal conductance parameters described in Lindroos
e.g. only NaF and Kv3 in the axon.
Also, Lindroos showed that Ca channel variation has very little effect on activity - no need to vary those for tuning to electrical activity - just for calcium concentration.
Lindroos also notes that NaF located in the soma and proximal dendrites are more easily phosphorylated
by PKA than the ones in the axon initial segment (Maurice et al., 2001; Hu et al., 2009).
Lots of results turning out burst.  Perhaps need another measure of burstiness?
'''

#from ajustador.helpers.copy_param.create_npz_param import create_npz_param
#npz_file='fitd1d2-D1-tmp_non05Jan2015_SLH00462938.npz'
#create_npz_param(npz_file, modeltype,ntype)
