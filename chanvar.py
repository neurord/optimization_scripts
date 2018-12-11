#from .sasparams file, print CV - std/mean to use for heterogeneity in network (multiplicative factor)
import numpy as np
import glob
files=['proto154F/fitFgp-proto-proto154.sasparams',
       'arky140F/fitgp-arky-arky140F.sasparams',
       'proto144F/fitFgp-proto-proto144.npz']
pattern='chan_proto*/*.sasparams'
files=glob.glob(pattern)

param_dict={}
for param_type in ['Cond','Chan']:
    param_set=[]
    for fn,fname in enumerate(files):
        print(fname)
        f=open(fname,'r')
        header=f.readline()
        items =header.split()
        param_list=[item.split('=') for item in items if item.startswith(param_type)]
        param_names = [param[0] for param in param_list]
        pvals=np.zeros(len(param_list))
        for i,param in enumerate(param_list):
            #print(i, param[0], param[1].split('+/-')[0],
            #      round(float(param[1].split('+/-')[1])/float(param[1].split('+/-')[0]),4))
            pvals[i]=param[1].split('+/-')[0]
        param_set.append(pvals)
    param_dict[param_type]={'values':np.array(param_set),'names':param_names}
    
for p in param_dict.values():
    for i,nm in enumerate(p['names']):
        prefix=''
        if nm.startswith('Chan'):
            if nm.endswith('vshift') and np.abs(np.mean(p['values'][:,i])) > np.std(p['values'][:,i]):
                prefix='***'
            if nm.endswith('taumul') and np.abs(np.mean(p['values'][:,i])-1.0) > np.std(p['values'][:,i]):
                prefix='***'
        print('{0} {1} {2} mean: {3:.5} stdev: {4:.5}'.format(prefix,nm,p['values'][:,i],
                                                              np.mean(p['values'][:,i]),np.std(p['values'][:,i])))

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

