import numpy as np
import glob
from matplotlib import pyplot as plt
import EPdata as epdata

plt.ion()
datadir='fitep-paramChan_120617_62938/tmp4a3mw9fe/'
datafiles=glob.glob('/tmp/'+datadir+'*.npy')

exp_to_fit=epdata.waves['120617'][[0, 2, 5]]
dt=exp_to_fit.waves[0].wave.x[1]

for wavenum in exp_to_fit.waves:
    plt.plot(wavenum.wave.x,wavenum.wave.y)

for df in datafiles:
    dat=np.load(df,'r')
    ts=np.arange(0,dt*len(dat),dt)
    plt.plot(ts,dat)

####### Create plot of fit history (panel 1) and model simulation (panel 2) for grant
    
    
import numpy as np
from matplotlib import pyplot as plt
import glob
import os

V2mV=1000
fsize=11

fitfile='opt1_1206172938/fitep-ep-tmp_1206172938.npz'
dat=np.load(fitfile)
fitness=dat['fitvals']

dirname='/home/avrama/moose/moose_nerp/moose_nerp/ep/'
fpattern='inject_p1b1b1'
suffix='Vm.txt'
tracefiles=glob.glob(dirname+fpattern+'*'+suffix)
traces=[];inject=[]
for tracef in tracefiles:
    traces.append(np.loadtxt(tracef,skiprows=0))
    inject.append(os.path.basename(tracef).split(fpattern)[1].split(suffix)[0])

plt.ion()
f = plt.figure(figsize=(7,2))
f.canvas.set_window_title('EP neuron optimization')
axes=f.add_subplot(121)
axes.plot(fitness[:,-1],'.')
axes.set_ylabel('fitness',fontsize = fsize)
axes.set_xlabel('model evaluation',fontsize = fsize)
axes.set_ylim(0,4.3)

axes=f.add_subplot(122)
for inj,trace in zip(inject,traces):
    axes.plot(trace[:,0],trace[:,1]*V2mV,label=inj)

axes=f.add_subplot(122)
for inj,trace in zip(inject,traces):
    axes.plot(trace[:,0],trace[:,1]*V2mV,label=inj)

axes.set_yticks([-150,-75,0])
axes.set_ylabel('Vm, mV',fontsize = fsize)   
axes.set_xlabel('Time, sec',fontsize = fsize)
#axes.legend(fontsize=10,loc='lower left')
plt.gcf().text(0.01,0.9,'A',fontsize=12)
plt.gcf().text(0.49,0.9,'B',fontsize=12)
f.tight_layout()

