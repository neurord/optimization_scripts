import ajustador as aju
from ajustador.helpers.copy_param import create_model_from_param as cmfp
import numpy as np
import os
import glob
import gpedata_experimental as gpe
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
def plot_traces(bestdir,dataname):
    exp_to_fit = gpe.alldata[dataname+'-2s'][[0,2,4]]
    traces=glob.glob(bestdir+'/*.npy')
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
target_dir='gp_opt/output/'
## Must specify morphfile if opt script specified morphfile different than what is in param_cond.py
morph= 'GP_soma.p'

os.chdir(target_dir)
pattern='cmaes*/fit*.npz'
fnames=glob.glob(pattern)

for npz_file in fnames:
    fitvals,tmpdirs=load_npz(npz_file)
    bestfit=np.argmin(fitvals[:,-1])
    bestdir=tmpdirs[bestfit]
    dataname=npz_file.split('cmaes_')[1].split('_')[0]
    neuron_type=npz_file.split('/')[1].split('-')[1]
    model=npz_file.split('/fit')[1].split('-')[0]

    plot_history(fitvals,dataname)
    plot_traces(bestdir,dataname)
    text=input('generate moose-nerp package (y/n)')
    if text=='n':
        print ("Next one")
    else:
        cmfp.createNewModelFolder(model, neuron_type, npz_file, dataname+'_1comp',morphfile=morph)
   
