import ajustador as aju
from ajustador.helpers import save_params,converge
import numpy as np
from ajustador import drawing
import measurements1 as ms1
import os

# a. simplest approach is to use CAPOOL (vs CASHELL, and CASLAB for spines)
# b. no spines
# c. use ghk (and ghkkluge=0.35e-6) once that is working/implemented in moose
ghkkluge=1

modeltype='squid'
rootdir='/home/avrama/moose/squid_opt'
#use 1 and 3 for testing, 250 and 8 for optimization
generations=200
popsiz=8
seed=62938
#after generations, do 25 more at a time and test for convergence
test_size=25

################## neuron /data specific specifications #############
ntype='D1'
dataname='D1_010612'
exp_to_fit = ms1.D1waves010612[[8,19,21,23]] #0, 6, 

dirname=dataname+str(seed)
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
os.chdir(rootdir+dirname)

tmpdir='/tmp/fit'+modeltype+'-'+ntype+'-'+dirname
######## set up parameters and fitness 
P = aju.optimize.AjuParam
params = aju.optimize.ParamSet(
    P('junction_potential', -.013,       fixed=1),
    P('RA',                 5.3,  min=1,      max=200),
    P('RM',                2.78,   min=0.1,      max=10),
    P('CM',                 0.010, min=0.001,      max=0.03),
    P('Eleak', -0.08, min=-0.080, max=-0.030),
    P('Cond_Na_0',      219e3,      min=0, max=600e3),
    P('Cond_Na_1',      1878,      min=0, max=10000),
    P('Cond_K', 1.7, min=0, max=5),
    P('Chan_Na_vshift_X',1.7, min=0, max=5),
    P('Chan_Na_vshift_Y',1.7, min=0, max=5),
    P('Chan_K_tau',1.7, min=0, max=5))
