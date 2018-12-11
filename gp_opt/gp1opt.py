import ajustador as aju
from ajustador.helpers import save_params,converge
from ajustador import drawing
import gpedata_experimental as gpe
import os
import params_fitness,fit_commands
########### Optimization of GP neurons ##############3
#proto 079, 154

ntype='proto'
modeltype='gp'
morph_file='GP1_41comp.p'
rootdir='/home/avrama/moose/gp_opt/'
#use 1 and 3 for testing, 200 and 8 for optimization
generations=200
popsiz=8
seed=62938
#after generations, do 25 more at a time and test for convergence
test_size=25

################## neuron /data specific specifications #############
dataname='proto079'
exp_to_fit = gpe.data[dataname+'-2s'][[0,2,4]]

dirname=dataname+'tmp_'+str(seed)
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
os.chdir(rootdir+dirname)

######## set up parameters and fitness to be used for all opts  ############
params1,fitness=params_fitness.params_fitness(morph_file,ntype,modeltype)

######## set-up and do the optimization  
fit1,mean_dict1,std_dict1,CV1=fit_commands.fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params1,generations,popsiz, seed, test_size)

###########look at results
drawing.plot_history(fit1, fit1.measurement)

#Save parameters of good results toward the end, and all fitness values
startgood=1500  #set to 0 to print/save all
threshold=0.40  #set to high value to print/save all
save_params.save_params(fit1, startgood, threshold)
#save_params.persist(fit1,'.')

################## Next neuron #############
'''dataname='proto154'
exp_to_fit = gpe.data[dataname+'-2s'][[0,2,4]]

dirname=dataname+'F_'+str(seed)
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
os.chdir(rootdir+dirname)

fit2,mean_dict2,std_dict2,CV2=fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params1,generations,popsiz, seed, test_size)

#look at results
drawing.plot_history(fit2, fit2.measurement)

threshold=0.34 
save_params.save_params(fit2, startgood, threshold) 
#save_params.persist(fit2,'.')
'''
