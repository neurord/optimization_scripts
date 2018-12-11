import ajustador as aju
from ajustador.helpers import save_params,converge
import numpy as np
from ajustador import drawing
import A2Acre as a2a
import os
#must be in current working directory for this import to work, else use exec
import params_fitness,fit_commands

# a. simplest approach is to use CAPOOL (vs CASHELL, and CASLAB for spines)
# b. no spines
# c. use ghk (and ghkkluge=0.35e-6) once that is working/implemented in moose
ghkkluge=1

modeltype='d1d2'
rootdir='/home/avrama/moose/SPN_opt/'
#use 1 and 3 for testing, 250 and 8 for optimization
generations=200
popsiz=8
seed=62938
#after generations, do 25 more at a time and test for convergence
test_size=25

################## neuron /data specific specifications #############
ntype='D1'
morph_file='MScell-primDend.p'
dataname='non05Jan2015_SLH004'
exp_to_fit = a2a.alldata[dataname][[0,20]] #0,6 are hyperpol

dirname='tmp_'+dataname+str(seed)
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
os.chdir(rootdir+dirname)

######## set up parameters and fitness 
params,fitness=params_fitness.params_fitness(morph_file,ntype,modeltype,ghkkluge)

########### set-up and do optimization
fit,mean_dict,std_dict,CV=fit_commands.fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params,generations,popsiz, seed, test_size)

#look at results
drawing.plot_history(fit, fit.measurement)

#Save parameters of good results from end of optimization, and all fitness values
startgood=0  #set to 0 to print all
threshold=5  #set to large number to print all

save_params.save_params(fit, startgood, threshold)

#to save the fit object
#save_params.persist(fit1,'.')
#import inspect

#for name, data in inspect.getmembers( fit):
#     print(name, data)
#from ajustador.helpers.save_param.copy_param import create_npz_param
#npz_file='fitd1d2-D1-tmp_non05Jan2015_SLH00462938.npz'
#create_npz_param(npz_file, modeltype,ntype)
