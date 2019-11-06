#from .sasparams file, print CV - std/mean to use for heterogeneity in network (multiplicative factor)
import numpy as np
import glob

#files=[]
pattern='gp_opt/pfc*proto*/*proto*.sasparams'
files=glob.glob(pattern)

param_dict={}
for param_type in ['Cond','Chan']:
    param_set=[]
    for fn,fname in enumerate(files):
        print(param_type,fname)
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
fitnum=10483
for p,val in fit1[fitnum].params.items():
   print(p,val)

to display a set of traces from optimization
a. determine tmp dir name (before exiting the optimization):
tmpdir=fit1[fitnum].tmpdir.name
b. 
import numpy as np
from matplotlib import pyplot

fnames=glob.glob(tmpdir+'/ivdata*.npy')
for fname in fnames:
  ivdata=np.load(fname,'r')
  tstop=0.7 (obtain from time value in waves)
  ts=np.linspace(0, tstop,num=len(ivdata),endpoint=False)
  pyplot.plot(ts,ivdata)

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

proto:
Chan_HCN1_taumul [ 1.9083   0.9187   0.95952  0.97779] mean: 1.1911 stdev: 0.41464
 Chan_HCN2_taumul [ 1.9905   0.59203  1.4825   0.91498] mean: 1.245 stdev: 0.5356
 Chan_HCN1_vshift [ -6.04600000e-05  -9.26590000e-03   1.27290000e-03   8.13060000e-03] mean: 1.9285e-05 stdev: 0.0061965
 Chan_HCN2_vshift [-0.0096722 -0.0099922 -0.0096786  0.0059477] mean: -0.0058488 stdev: 0.006812
*** Chan_NaF_vshift [ -3.34750000e-03   4.06940000e-05  -6.72880000e-03  -9.71050000e-03] mean: -0.0049365 stdev: 0.0036504
 Chan_NaF_taumul [ 0.61506  0.97911  0.90506  0.94871] mean: 0.86199 stdev: 0.14497
 Chan_NaS_vshift [ 0.0094233  -0.00073046  0.0060544  -0.0011393 ] mean: 0.003402 stdev: 0.0044998
*** Chan_NaS_taumul [ 1.4676  1.4572  1.7104  1.195 ] mean: 1.4576 stdev: 0.18232
 Chan_Kv3_vshift [-0.0018977  0.0099984  0.0042866 -0.0049803] mean: 0.0018517 stdev: 0.005767
*** Chan_Kv3_taumul [ 1.8573  1.4688  1.7181  1.1637] mean: 1.552 stdev: 0.26386
 Chan_KvS_vshift [-0.0059736 -0.0021406  0.003426  -0.0067433] mean: -0.0028579 stdev: 0.0040251
 Chan_KvS_taumul [ 0.72348  0.6798   1.4679   1.2446 ] mean: 1.0289 stdev: 0.33705
 Chan_KvF_vshift [-0.0022336  0.0017756  0.0076543 -0.0071204] mean: 1.8975e-05 stdev: 0.0054182
*** Chan_KvF_taumul [ 1.7391  1.6428  1.7907  1.76  ] mean: 1.7331 stdev: 0.055298

arky:
 Chan_HCN1_taumul [ 1.845    0.58781  0.59823] mean: 1.0103 stdev: 0.5902
 Chan_HCN2_taumul [ 0.60959  1.1425   1.427  ] mean: 1.0597 stdev: 0.3388
 Chan_HCN1_vshift [-0.0013654  0.0093422  0.0014421] mean: 0.0031396 stdev: 0.0045332
 Chan_HCN2_vshift [-0.00085856 -0.0099948   0.0019554 ] mean: -0.002966 stdev: 0.0051012
 Chan_NaF_vshift [ 0.0050508 -0.0036798 -0.0099941] mean: -0.0028744 stdev: 0.0061684
 Chan_NaF_taumul [ 0.56226  0.60678  1.1252 ] mean: 0.76475 stdev: 0.25553
 Chan_NaS_vshift [ 0.0015642  0.0035505 -0.005057 ] mean: 1.9233e-05 stdev: 0.0036799
 Chan_NaS_taumul [ 1.6733   0.61967  1.7037 ] mean: 1.3322 stdev: 0.504
*** Chan_Kv3_vshift [ 0.0050261  0.0094099  0.0099836] mean: 0.0081399 stdev: 0.0022142
 Chan_Kv3_taumul [ 0.88784  1.392    1.8515 ] mean: 1.3771 stdev: 0.39355
 Chan_KvS_vshift [ 0.0057974  0.0022875 -0.0040061] mean: 0.0013596 stdev: 0.0040557
 Chan_KvS_taumul [ 1.5269   0.71419  1.7049 ] mean: 1.3153 stdev: 0.43124
 Chan_KvF_vshift [-0.002021   0.0068487  0.0072296] mean: 0.0040191 stdev: 0.0042738
 Chan_KvF_taumul [ 1.4154   0.84545  0.62516] mean: 0.962 stdev: 0.33297

Just set of good proto neurons:
 Chan_HCN1_taumul [ 1.9083  0.9187] mean: 1.4135 stdev: 0.4948
 Chan_HCN2_taumul [ 1.9905   0.59203] mean: 1.2913 stdev: 0.69923
*** Chan_HCN1_vshift [ -6.04600000e-05  -9.26590000e-03] mean: -0.0046632 stdev: 0.0046027
*** Chan_HCN2_vshift [-0.0096722 -0.0099922] mean: -0.0098322 stdev: 0.00016
 Chan_NaF_vshift [ -3.34750000e-03   4.06940000e-05] mean: -0.0016534 stdev: 0.0016941
*** Chan_NaF_taumul [ 0.61506  0.97911] mean: 0.79709 stdev: 0.18202
 Chan_NaS_vshift [ 0.0094233  -0.00073046] mean: 0.0043464 stdev: 0.0050769
*** Chan_NaS_taumul [ 1.4676  1.4572] mean: 1.4624 stdev: 0.0052
 Chan_Kv3_vshift [-0.0018977  0.0099984] mean: 0.0040503 stdev: 0.005948
*** Chan_Kv3_taumul [ 1.8573  1.4688] mean: 1.6631 stdev: 0.19425
*** Chan_KvS_vshift [-0.0059736 -0.0021406] mean: -0.0040571 stdev: 0.0019165
*** Chan_KvS_taumul [ 0.72348  0.6798 ] mean: 0.70164 stdev: 0.02184
 Chan_KvF_vshift [-0.0022336  0.0017756] mean: -0.000229 stdev: 0.0020046
*** Chan_KvF_taumul [ 1.7391  1.6428] mean: 1.6909 stdev: 0.04815

set of good neurons:
 Chan_HCN1_taumul [ 1.9083   0.9187   0.59823  1.845    0.58781] mean: 1.1716 stdev: 0.58817
 Chan_HCN2_taumul [ 1.9905   0.59203  1.427    0.60959  1.1425 ] mean: 1.1523 stdev: 0.52661
 Chan_HCN1_vshift [ -6.04600000e-05  -9.26590000e-03   1.44210000e-03  -1.36540000e-03
   9.34220000e-03] mean: 1.8508e-05 stdev: 0.0059511
*** Chan_HCN2_vshift [-0.0096722  -0.0099922   0.0019554  -0.00085856 -0.0099948 ] mean: -0.0057125 stdev: 0.0051902
 Chan_NaF_vshift [ -3.34750000e-03   4.06940000e-05  -9.99410000e-03   5.05080000e-03
  -3.67980000e-03] mean: -0.002386 stdev: 0.0049331
 Chan_NaF_taumul [ 0.61506  0.97911  1.1252   0.56226  0.60678] mean: 0.77768 stdev: 0.22952
 Chan_NaS_vshift [ 0.0094233  -0.00073046 -0.005057    0.0015642   0.0035505 ] mean: 0.0017501 stdev: 0.0047884
 Chan_NaS_taumul [ 1.4676   1.4572   1.7037   1.6733   0.61967] mean: 1.3843 stdev: 0.39559
*** Chan_Kv3_vshift [-0.0018977  0.0099984  0.0099836  0.0050261  0.0094099] mean: 0.0065041 stdev: 0.0045942
*** Chan_Kv3_taumul [ 1.8573   1.4688   1.8515   0.88784  1.392  ] mean: 1.4915 stdev: 0.35728
 Chan_KvS_vshift [-0.0059736 -0.0021406 -0.0040061  0.0057974  0.0022875] mean: -0.00080708 stdev: 0.0042872
 Chan_KvS_taumul [ 0.72348  0.6798   1.7049   1.5269   0.71419] mean: 1.0699 stdev: 0.44962
 Chan_KvF_vshift [-0.0022336  0.0017756  0.0072296 -0.002021   0.0068487] mean: 0.0023199 stdev: 0.0041107
 Chan_KvF_taumul [ 1.7391   1.6428   0.62516  1.4154   0.84545] mean: 1.2536 stdev: 0.44156
'''
