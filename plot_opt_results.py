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
    return fitvals,tmpdirs

def plot_history(fitvals,dataname):
    plt.figure()
    plt.plot(fitvals[:,-1],'r.')
    plt.title(dataname)
    plt.xlabel('evaluation')
    plt.ylabel('fitness')
    return

def plot_traces(bestdir,dataname,traces,exp_to_fit):
    plt.figure()
    for wave in exp_to_fit.waves:
        plt.plot(wave.wave.x,wave.wave.y)
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
###### Specify directory that has output directory with cmaes/fit*.npz files in it.
'''
target_dir='gp_opt/output/'
wave_module='gpedata_experimental'
wave_set=[0,2,4]

target_dir='Str_opt/output'
wave_module='A2Acre'
wave_set={'FSI':[1, 5, 14, 17, 20], 'D1': [1, 5, 14, 17, 22], 'D2': [1, 5, 12, 15, 19]}
'''
target_dir='epopt-all/output'
wave_module='EPdata'
wave_set=[0,2,5,7]

#different data naming conventions for different wave modules
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
    fitvals,tmpdirs=load_npz(npz_file)
    bestfit=np.argmin(fitvals[:,-1])
    bestdir=tmpdirs[bestfit]
    model=npz_file.split('/fit')[1].split('-')[0]
    neuron_type=npz_file.split('/fit')[1].split('-')[1]
    dataname=npz_file.split('cmaes_')[1].split('_')[0]
    if isinstance(wave_set,list):
        exp_to_fit = wavedir.alldata[dataname+suffix][wave_set]
    elif isinstance(wave_set,dict):
        exp_to_fit = wavedir.alldata[dataname+suffix][wave_set[neuron_type]]
    else:
        print('create new condition for specifying list of waves')
    plot_history(fitvals,dataname)
    traces=glob.glob(bestdir+'/*.npy')
    print('num traces',len(traces), 'num waves', len(exp_to_fit))
    if len(traces):
        plot_traces(bestdir,dataname,traces,exp_to_fit)
        text=input('generate moose-nerp package (y/n)')
        if text=='n':
            print ("Next one")
        else:
            cmfp.createNewModelFolder(model, neuron_type, npz_file, dataname+'_1comp',morphfile=morph,copy_traces=True)
    else:
        print('>>>> fit directorywith traces {} does not exist'.format(dataname))
   
