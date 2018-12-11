import ajustador as aju
from ajustador.helpers import save_params
from ajustador import drawing
import measurements1 as ms1
import os
#must be in current working directory for this import to work, else use exec
import params_fitness,fit_commands

# a. simplest approach is to use CAPOOL (vs CASHELL, and CASLAB for spines)
# b. no spines
# c. use ghk (and ghkkluge=0.35e-6) once that is working/implemented in moose
ghkkluge=1

modeltype='d1d2'
rootdir='/home/avrama/moose/SPN_opt/'
#use 1 and 3 for testing, 200 and 8 for optimization
generations=200
popsiz=8
seed=62938
#after generations, do 25 more at a time and test for convergence
test_size=25

################## neuron /data specific specifications #############
ntype='D2'
dataname='D2_010612'
exp_to_fit = ms1.D2waves010612[[8, 18,20,22]] #0, 6, 

dirname=dataname+'_pas2_'+str(seed)
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
os.chdir(rootdir+dirname)

######## set up parameters and fitness 
params1,fitness=params_fitness.params_fitness(morph_file,ntype,modeltype)

# set-up and do optimization
fit3,mean_dict3,std_dict3,CV3=fit_commands.fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params1,generations,popsiz, seed, test_size)

#########look at results
drawing.plot_history(fit3, fit3.measurement)

#Save parameters of good results from end of optimization, and all fitness values
startgood=1000  #set to 0 to print all
threshold=0.8  #set to large number to print all
save_params.save_params(fit3, startgood, threshold)

#to save the fit object
#save_params.persist(fit3,'.')

