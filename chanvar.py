 #from .sasparams file, print CV - std/mean to use for heterogeneity in network (multiplicative factor)
import numpy as np
import glob
import sys
import pandas as pd
import anal_util as au

#files=[]
#Enter the path to files
results_dir=sys.argv[1]
neurtypes=sys.argv[2].split()
#syntax: python3 -i opt_scripts/chanvar.py dirname 'proto Npas'
pattern=results_dir+'*/*.npz'
files=sorted(glob.glob(pattern))
group_names={n:[f for f in files if n in f] for n in neurtypes}
tile=0.005
num_fits=5
param_types=['Cond','Chan']

df_list = {}
df_list_of_lists = {} 
for neur in neurtypes:
    df_list[neur], df_list_of_lists[neur] = au.combined_df(group_names[neur], tile, neur,0)

min_samples = np.min([n.shape[0] for vals in df_list_of_lists.values() for n in vals])
num_samples=min(min_samples, num_fits)

smalldf_list = {neur:[] for neur in neurtypes}
for neur in neurtypes:
    for i in range(len(df_list_of_lists[neur])):
        smalldf_list[neur].append(df_list_of_lists[neur][i][-num_samples:])

alldf=pd.concat([df for dfset  in smalldf_list.values() for df in dfset])
column_list=[col for par in param_types for col in list(alldf.columns) if col.startswith(par) ]+['RA','CM','RM']
neuron_pars=alldf.groupby(['neuron','cell'])[column_list].mean()

param_dict={}
for col in neuron_pars.columns:
    mean_dict=neuron_pars.groupby(['neuron'])[col].mean().to_dict()
    stdvals=list(neuron_pars.groupby(['neuron'])[col].std())
    param_dict[col]={neur:(m,s) for (neur,m),s in zip(mean_dict.items(),stdvals)}
    param_dict[col]['all']=(neuron_pars[col].mean(),neuron_pars[col].std())

for par,vals in param_dict.items():
    for neur,(mn,sd) in vals.items():
        prefix=''
        if par.startswith('Chan'):
            if par.endswith('vshift') and (np.abs(mn) > sd):
                prefix='***'
            if par.endswith('taumul') and (np.abs(mn-1.0) > sd):
                prefix='***'
        print('{0} {1} {2} mean: {3:.5} stdev: {4:.5}'.format(prefix,par,neur,mn,sd))
    if len(neurtypes)==2:
        if (vals[neurtypes[0]][0]-vals[neurtypes[1]][0])/np.mean([vals[neurtypes[0]][1],vals[neurtypes[1]][1]])>2:
            print('### sig diff {}'.format(par))

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
fitnum=10483
for p,val in fit1[fitnum].params.items():
   print(p,val)

'''

'''channel parameters are not the same for all GP neurons, not even for all GP/arky or GP/proto
Several questions/analyses are needed
1. for small vshift (<2 mV) or taumult ~1 (e.g. 0.9 or 1.1) ignoring these parameters produces minimal change
        Question: what counts as small?  I.e., 3 mV?  0.8 or 1.2?
        partly address with simulations - ignore deltas - visually comare neurons
2. which parameters are similar, and which don't match within a neuron class. 
        Question: can parameters that vary be ignored (related to Q1) 
        calculate mean values to quantify the difference  - mostly mean < stdev
        partly address with simulations - ignore detlas for all but these?
                                        - repeat optimizations allowing only those channel parameters to vary?
3. Once identify parameters for a class that are critical (given conductances):
        Question: can optimization be done to find new conductances?
        Is answer above yes within class but no between class?

'''
