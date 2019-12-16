import ajustador as aju
import numpy as np
import measurements1 as ms1
import gpedata_experimental as gpe

wave_set=[ ms1.D1waves010612[19],ms1.D1waves010612[21],ms1.D1waves010612[23],
           ms1.D1waves042811[20],ms1.D1waves042811[22],ms1.D1waves042811[23],
           ms1.D1waves051811[20],ms1.D1waves051811[22],ms1.D1waves051811[23],
           ms1.D2waves010612[18],ms1.D2waves010612[20],ms1.D2waves010612[22],
           ms1.D2waves051311[17],ms1.D2waves051311[19],ms1.D2waves051311[22],
           ms1.D2waves081011[15],ms1.D2waves081011[17],ms1.D2waves081011[19]]

hyper_set=[ ms1.D1waves010612[8],
            ms1.D1waves042811[8],
            ms1.D1waves051811[8],
            ms1.D2waves010612[8],
            ms1.D2waves051311[8],
            ms1.D2waves081011[8]]

for wave in wave_set:
    ahp_amp=aju.features.AHP(wave).spike_ahp.x-aju.features.Spikes(wave).spike_threshold
    print(wave.filename[0:10], np.round(wave.injection*1e9,4),
          aju.features.Spikes(wave).spike_count,
          np.round(aju.features.Spikes(wave).spike_height.mean(),4),
          np.round(aju.features.Spikes(wave).spike_width.mean(),5),
          aju.features.Spikes(wave).spike_latency,
          np.round(ahp_amp.mean(),4) ,
          np.round(aju.features.SteadyState(wave).baseline_pre.x,4),
          np.round(aju.features.SteadyState(wave).baseline_post.x,4),
          np.round(aju.features.SteadyState(wave).response.x,4),
          np.round(aju.features.FallingCurve(wave).falling_curve_tau.x,5),
          np.round(aju.features.ChargingCurve(wave).charging_curve_halfheight.x,4))

########### GPE data
wave_set=[gpe.data['proto079-2s'][[0,2,4]],
          gpe.data['proto154-2s'][[0,2,4]],
          gpe.data['proto144-2s'][[0,2,4]],
          gpe.data['proto122-2s'][[0,2,4]],
          gpe.data['arky120-2s'][[0,2,4]],
          gpe.data['arky138-2s'][[0,2,4]],
          gpe.data['arky140-2s'][[0,2,4]]]

for wave in wave_set:
    for trace in wave:
        ahp_amp=aju.features.AHP(trace).spike_ahp.x-aju.features.Spikes(trace).spike_threshold
        print(wave.name, np.round(trace.injection*1e9,4),
              aju.features.Spikes(trace).spike_count,
              np.round(aju.features.Spikes(trace).spike_height.mean(),4),
              np.round(aju.features.Spikes(trace).spike_width.mean(),5),
              np.round(ahp_amp.mean(),4) ,
              np.round(aju.features.SteadyState(trace).baseline_post.x,4),
              np.round(aju.features.Rectification(trace).rectification.x,4),
              np.round(aju.features.SteadyState(trace).response.x,4),
              np.round(aju.features.FallingCurve(trace).falling_curve_tau.x,5),
              )

        

    '''
   AHP curve: this is direct comparison between measurement and simulation, so there is no measure for one or the other.  Could compare AHP position, but since not used in optimization, will not report
spike_ahp_position is absolute (relative to all time) thus need to substract spike position
Do not use until verify that these values are correct
'''
for wave in wave_set:
    ahp_time=aju.features.AHP(wave).spike_ahp_position.x-aju.features.Spikes(wave).spikes.x
    print(wave.filename[0:10], np.round(wave.injection*1e9,4), ahp_time.mean())
