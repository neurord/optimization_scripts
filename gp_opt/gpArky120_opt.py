import ajustador as aju
from ajustador.helpers import save_params,converge
from ajustador import drawing
import gpedata_experimental as gpe
import os
#must be in current working directory for this import to work, else use exec
import params_fitness as pfc
import fit_commands

########### Optimization of GP neurons ##############3
modeltype='gp'
rootdir='/home/avrama/moose/opt_scripts/gp_opt/'
#use 1 and 3 for testing, 200 and 8 for optimization
generations=100
popsiz=8
seed=938
#after generations, do 25 more at a time and test for convergence
test_size=25

################## neuron /data specific specifications #############
ntype='arky'
morph_file='GP_arky_41comp.p'
dataname='arky120'
exp_to_fit = gpe.data[dataname+'-2s'][[0,2,4]]

dirname='cond_'+dataname+str(seed)
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
os.chdir(rootdir+dirname)

######## set up parameters and fitness to be used for all opts  ############
params1,fitness=pfc.params_fitness(morph_file,ntype,modeltype)

# set-up and do the optimization 
fit1=fit_commands.fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params1,generations,popsiz, seed, test_size)
if test_size>0:
    mean_dict,std_dict,CV=converge.iterate_fit(fit1,test_size,popsiz,std_crit=0.02,max_evals=12000)

###########look at results
drawing.plot_history(fit1, fit1.measurement)

#Save parameters of good results toward the end, and all fitness values
startgood=0  #set to 0 to print/save all
threshold=10  #set to high value to print/save all

save_params.save_params(fit1, startgood, threshold)
#save_params.persist(fit1,'.')

'''
#to create moose_nerp parameter file:
from ajustador.helpers.copy_param import create_npz_param
npzfile=fit1.name+'.npz'
create_npz_param.create_npz_param(npzfile,modeltype,ntype)

#to print params of centroid
for nm,val,stdev in zip(fit1.param_names(),
                                    fit1.params.unscale(fit1.optimizer.result()[0]),
                                    fit1.params.unscale(fit1.optimizer.result()[6])):
  print(nm,'=', val,'+/-', stdev)

to print parameters of particular fit:
fitnum=7701
for p,val in fit1[fitnum].params.items():
   print(p,val)

'''
