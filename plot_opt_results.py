import ajustador as aju
from ajustador.helpers.copy_param import create_model_from_param as cmfp
import numpy as np
import os
import glob
import importlib
from matplotlib import pyplot as plt
plt.ion()

def load_npz(fname):
    dat = np.load(fname)
    fitvals=dat['fitvals']
    tmpdirs=dat['tmpdirs']
    paramsdir=dat['params']
    paramnames=dat['paramnames']
    return fitvals,tmpdirs,paramsdir,paramnames

def plot_history(fitvals,dataname):
    plt.figure()
    plt.plot(fitvals[:,-1],'r.')
    plt.title(dataname)
    plt.xlabel('evaluation')
    plt.ylabel('fitness')
    return

def plot_traces(bestdir,dataname,traces,exp_to_fit,jp=0):
    plt.figure()
    for wave in exp_to_fit.waves:
        plt.plot(wave.wave.x,wave.wave.y+jp)
    maxtime=wave.wave.x[-1]
    for trace_file in traces:
        trace=np.load(trace_file)
        time=np.linspace(0,maxtime,num=len(trace),endpoint=True)
        plt.plot(time,trace)
    plt.title(dataname)
    plt.xlabel('Time')
    plt.ylabel('Vm')
    return

#################### main ############
#target_dir: directory that has output directory with cmaes/fit*.npz
#wave_module: module in python package waves with experimental data
#wave_set: subset of waves used idn optimization

target_dir='Proto154_84362/optimization_scripts/output/'
opt_path='../moose_nerp'
##relative to target_dir
wave_module='gpedata_experimental'
wave_set=[0,2,4]
'''
target_dir='Str_opt/output'
wave_module='A2Acre'
wave_set={'FSI':[1, 5, 14, 17, 20], 'D1': [1, 5, 14, 17, 22], 'D2': [1, 5, 12, 15, 19]}
'''
'''
target_dir='epopt-all/output'wave_module='EPdata'
wave_set=[0,2,5,7]
'''
#different data naming conventions for different wave modules
#this part still needs to be customized according to how npz files were named
#this is used to determine name of experimental data from name of npz_file
#clearly it would be better to save data name in npz file
if wave_module.startswith('A2A'):
    suffix='_'+npz_file.split('cmaes_')[1].split('_')[1]
elif wave_module.startswith('gp'):
    suffix='-2s'
else:
    suffix=''
## Must specify morphfile if opt script specified morphfile different than what is in param_cond.py
morph= None

wavedir=importlib.import_module(wave_module)
os.chdir(target_dir)
pattern='cmaes*/fit*.npz'
fnames=glob.glob(pattern)

for npz_file in fnames:
    fitvals,tmpdirs,params,paramnames=load_npz(npz_file)
    jp_index=list(paramnames).index('junction_potential')
    model=npz_file.split('/fit')[1].split('-')[0]
    neuron_type=npz_file.split('/fit')[1].split('-')[1]
    dataname=npz_file.split('cmaes_')[1].split('_')[0]
    if isinstance(wave_set,list):
        exp_to_fit = wavedir.alldata[dataname+suffix][wave_set]
        print(exp_to_fit)
    elif isinstance(wave_set,dict):
        exp_to_fit = wavedir.alldata[dataname+suffix][wave_set[neuron_type]]
    else:
        print('create new condition for specifying list of waves')
    plot_history(fitvals,dataname)
    bestfit=np.argmin(fitvals[:,-1])
    while bestfit>0:
        bestdirnsg=tmpdirs[bestfit]
        bestdir='/'.join(bestdirnsg.split('/')[10:])
        bestfitval=fitvals[bestfit,-1]
        jpval=params[bestfit][jp_index]
        print(jpval)
        traces=glob.glob(bestdir+'/*.npy')
        print('num traces',len(traces), 'num waves', len(exp_to_fit),'bestfit number=',bestfit)
        if len(traces)>0:
            plot_traces(bestdir,dataname,traces,exp_to_fit, jp=jpval)
            previous_bestfit=bestfit
            text=input('do you want to try another fit (fitnumber for yes /0 for no)')
            bestfit=int(text)
    bestfit=previous_bestfit
    text=input('generate moose-nerp package (y/n)')
    if text=='n':
        print ("Next one")
    elif len(traces):
        cmfp.createNewModelFolder(model, neuron_type, npz_file, dataname,morphfile=morph,copy_traces=True,bestfit=bestfit)
    else:
        print('>>>> fit directorywith traces {} does not exist'.format(dataname))
        cmfp.createNewModelFolder(model, neuron_type, npz_file, dataname,morphfile=morph,copy_traces=False,bestfit=bestfit)
