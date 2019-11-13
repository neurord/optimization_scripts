import ajustador as aju
from ajustador.helpers import save_params,converge
from ajustador import drawing
import os
import EPdata as epdata
import params_fitness_ep as pf
import fit_commands as fc

modeltype='ep'
rootdir=os.getcwd()+'/'
generations=100
popsiz=8
seed=162938
#after generations, do 25 more at a time and test for convergence
test_size=25

################## neuron /data specific specifications #############
dataname='051517'
ntype='ep'
morph_file='EP_93comp.p'
exp_to_fit=epdata.waves[dataname][[0, 2, 5, 7]]

dirname='pchan_'+dataname+'_'+str(seed)
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
os.chdir(rootdir+dirname)

######## set up parameters and fitness to be used for all opts  ############
params1,fitness=pf.params_fitness(morph_file,ntype,modeltype)

######## set-up and do the optimization 
fit1=fc.fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params1,generations,popsiz, seed)

if test_size>0:
    mean_dict1,std_dict1,CV1=converge.iterate_fit(fit1,test_size,popsiz,std_crit=0.02,max_evals=12000)

###########look at results
drawing.plot_history(fit1, fit1.measurement)

startgood=0  #set to 0 to print all
threshold=10  #set to large number to print all

save_params.save_params(fit1, startgood, threshold)
'''Sag and spike times match great, but AHPs way to big.  
Perhaps need another channel?  Is a second KA current needed?
Could KA replace KDr?  Check the "optimal" conductance
Would additional calcium currents help?  What is known about them?
'''
