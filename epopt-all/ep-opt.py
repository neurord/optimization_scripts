import ajustador as aju
from ajustador.helpers import save_params,converge
from ajustador import drawing
import os
import EPdata as epdata

modeltype='ep'
rootdir='/home/avrama/moose/opt_scripts/epopt-all/'
generations=100
popsiz=8
seed=162938
#after generations, do 25 more at a time and test for convergence
test_size=25

################## neuron /data specific specifications #############
dataname='120617'
ntype='ep'
morph_file='EP_41compA.p'
exp_to_fit=epdata.waves['120617'][[0, 2, 5]]

dirname='ChanKaF_'+dataname+'_'+str(seed)
if not dirname in os.listdir(rootdir):
    os.mkdir(rootdir+dirname)
os.chdir(rootdir+dirname)

######## set up parameters and fitness   ############
P = aju.optimize.AjuParam
fitness = aju.fitnesses.combined_fitness('empty',
                                         response=1,
                                         baseline_pre=1,
                                         baseline_post=1,
                                         rectification=1,
                                         falling_curve_time=1,
                                         spike_time=1,
                                         spike_width=1,
                                         spike_height=1,
                                         spike_latency=1,
                                         spike_count=1,
                                         spike_ahp=1,
                                         ahp_curve=1,
                                         spike_range_y_histogram=1)

############## Params 3: allow HCN, and ca in axon initial segment AND change in channel parameters ###################

params4 = aju.optimize.ParamSet(
    P('junction_potential', -0.012, min=-0.020, max=-0.005),
    P('RA',                 5 ,     min=0.1, max=10),
    P('RM',                 7,      min=0.1,  max=12),
    P('CM',                 0.01,   min=.003, max=0.03),
    P('Eleak', -0.031, min=-0.070, max=-0.010),
    P('Cond_KDr_0', 6, min=0, max=100),
    P('Cond_KDr_1', 6, min=0, max=30),
    P('Cond_KDr_2', 366, min=0, max=1000),
    P('Cond_Kv3_0', 60, min=0, max=3000),
    P('Cond_Kv3_1', 273, min=0, max=3000),
    P('Cond_Kv3_2', 742, min=0, max=6000),
    P('Cond_KvS_0',  26, min=0, max=200),
    P('Cond_KvS_1',  5, min=0, max=200),
    P('Cond_KvS_2',  31, min=0, max=200),
    P('Cond_KvF_0',  10, min=0, max=50),
    P('Cond_KvF_1',  10, min=0, max=50),
    P('Cond_KvF_2',  10, min=0, max=100),
    P('Cond_NaF_0', 621, min=00, max=2000),
    P('Cond_NaF_1', 84, min=0, max=500),
    P('Cond_NaF_2', 931, min=00, max=20000),
    P('Cond_NaS_0', 6, min=0, max=10),
    P('Cond_NaS_1', 2.6, min=0, max=10),
    P('Cond_NaS_2', 9.8, min=0, max=10),
    P('Cond_HCN1_0', 0.5, min=0.0, max=4),
    P('Cond_HCN1_1', 0.01, min=0.0, max=4),
    P('Cond_HCN2_0', 3, min=0.0, max=10),
    P('Cond_HCN2_1', 0.7, min=0.0, max=4),
    P('Cond_Ca_0', 0.045, min=0, max=1),
    P('Cond_Ca_1', 0.09, min=0, max=1),
    P('Cond_SKCa_0', 6, min=0, max=20),
    P('Cond_SKCa_1', 4, min=0, max=20),
    P('Cond_BKCa_0', 0.5, min=0, max=10),
    P('Cond_BKCa_1', 3.8, min=0, max=10),
    P('Chan_HCN1_taumul', 1.0, min=0.4, max=2.4),
    P('Chan_HCN1_vshift', 0.0, min=-0.01, max=0.01),
    P('Chan_HCN2_taumul', 1.0, min=0.4, max=2.4),
    P('Chan_HCN2_vshift', 0.0, min=-0.01, max=0.01),
    P('Chan_NaF_vshift', 0.0, min=-0.01, max=0.01),
    P('Chan_NaF_taumul', 1.0, min=0.6, max=1.8),
    P('Chan_NaS_vshift', 0.0, min=-0.01, max=0.01),
    P('Chan_NaS_taumul', 1.0, min=0.6, max=1.8),
    P('Chan_Kv3_vshift', 0.0, min=-0.01, max=0.01),
    P('Chan_Kv3_taumul', 1.0, min=0.6, max=1.8),
    P('Chan_KvS_vshift', 0.0, min=-0.01, max=0.01),
    P('Chan_KvS_taumul', 1.0, min=0.4, max=2.4),
    P('Chan_KvF_vshift', 0.0, min=-0.01, max=0.01),
    P('Chan_KvF_taumul', 1.0, min=0.4, max=2.4),
    P('morph_file', morph_file, fixed=1),
    P('neuron_type',     ntype, fixed=1),
    P('model',           modeltype,     fixed=1))

tmpdir='/tmp/fit'+modeltype+'-'+dirname
fit4 = aju.optimize.Fit(tmpdir,
                          exp_to_fit,
                          modeltype, ntype,
                          fitness, params4,
                          _make_simulation=aju.optimize.MooseSimulation.make,
                          _result_constructor=aju.optimize.MooseSimulationResult)

fit4.load()
fit4.do_fit(generations, popsize=popsiz,seed=seed)
if test_size>0:
    mean_dict4,std_dict4,CV4=converge.iterate_fit(fit4,test_size,popsiz,max_evals=8000)
#
drawing.plot_history(fit4, fit4.measurement)

startgood=0  #set to 0 to print all
threshold=10  #set to large number to print all

save_params.save_params(fit4, startgood, threshold)
'''Sag and spike times match great, but AHPs way to big.  
Perhaps need another channel?  Is a second KA current needed?
Could KA replace KDr?  Check the "optimal" conductance
Would additional calcium currents help?  What is known about them?

observation: strong Kv3 associated with nice looking AHPs, and as it decreases AHPs look worse
------>>>> evaluate Kv3 inactivation - too strong?
     spike height decreases and NaS decreases along with AHP decrease.  Symptom or cause?
    Z gate of NaS decreases slowly across simulation - this could be causal, BUT
    eliminating NaS Zgate doesn't change this.
no change Ca current, SKCa, HCN1/2, BK. KvS has slow decay - not corr with peak
KDr peak seems correlated with AHP - check on gating - if necessary eliminate Y
Is KvF needed for fast AHP?

RESULTS:fitep-ChanKaF_120617_162938
converged with m=0.39, but still not great - too many spikes, wrong shape AHP
Dan is working on fixing the fit to AHP amplitude.  Repeat optimization once that is fixed.
'''
