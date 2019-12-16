import ajustador as aju

def params_fitness(morph_file,ntype,modeltype):

    P = aju.optimize.AjuParam
    params = aju.optimize.ParamSet(
        P('junction_potential', -0.012, min=-0.020, max=-0.005),
        P('RA',                 2,     min=0.1, max=12),
        P('RM',                 2,      min=0.1,  max=12),
        P('CM',                 0.014,   min=0.005, max=0.03),
        P('Eleak', -0.040, min=-0.070, max=-0.020),
        P('Cond_KDr_0', 40, min=0, max=1000),
        P('Cond_Kv3_0', 366, min=0, max=1000),
        P('Cond_KvF_0',  10, min=0, max=50),
        P('Cond_KvS_0', 1, min=0, max=50),
        P('Cond_NaF_0', 400, min=100, max=100e3),
        P('Cond_HCN1_0', 0.2, min=0.0, max=5),
        P('Cond_HCN2_0', 0.2, min=0.0, max=5),
        P('Cond_NaS_0', 0.5, min=0, max=10),
        P('Cond_Ca_0', 0.1, min=0, max=2),
        P('Cond_SKCa_0', 2, min=0, max=100),
        P('Cond_BKCa_0', 2, min=0, max=800),
        P('Chan_HCN1_taumul', 1.0, min=0.5, max=2),
        P('Chan_HCN2_taumul', 1.0, min=0.5, max=2),
        P('Chan_HCN1_vshift', 0.0, min=-0.01, max=0.01),
        P('Chan_HCN2_vshift', 0.0, min=-0.01, max=0.01),
        P('Chan_NaF_vshift_X', 0.0, min=-0.01, max=0.01),
        P('Chan_NaF_vshift_Y', 0.0, min=-0.01, max=0.01),
        P('Chan_NaF_taumul_X', 1.0, min=0.5, max=2),
        P('Chan_NaF_taumul_Y', 1.0, min=0.5, max=2),
        P('Chan_NaS_vshift_X', 0.0, min=-0.01, max=0.01),
        P('Chan_NaS_vshift_Y', 0.0, min=-0.01, max=0.01),
        P('Chan_NaS_taumul_X', 1.0, min=0.5, max=2),
        P('Chan_NaS_taumul_Y', 1.0, min=0.5, max=2),
        P('Chan_Kv3_vshift_X', 0.0, min=-0.01, max=0.01),
        P('Chan_Kv3_vshift_Y', 0.0, min=-0.01, max=0.01),
        P('Chan_Kv3_taumul_X', 1.0, min=0.5, max=2),
        P('Chan_Kv3_taumul_Y', 1.0, min=0.5, max=2),
        P('Chan_KvS_vshift_X', 0.0, min=-0.01, max=0.01),
        P('Chan_KvS_vshift_Y', 0.0, min=-0.01, max=0.01),
        P('Chan_KvS_taumul_X', 1.0, min=0.6, max=1.8),
        P('Chan_KvS_taumul_Y', 1.0, min=0.6, max=1.8),
        P('Chan_KvF_vshift_X', 0.0, min=-0.01, max=0.01),
        P('Chan_KvF_vshift_Y', 0.0, min=-0.01, max=0.01),
        P('Chan_KvF_taumul_X', 1.0, min=0.6, max=1.8),
        P('Chan_KvF_taumul_Y', 1.0, min=0.6, max=1.8),
        P('Chan_KDr_vshift_X', 0.0, min=-0.01, max=0.01),
        P('Chan_KDr_vshift_Y', 0.0, min=-0.01, max=0.01),
        P('Chan_KDr_taumul_X', 1.0, min=0.6, max=1.8),
        P('Chan_KDr_taumul_Y', 1.0, min=0.6, max=1.8),
        P('morph_file', morph_file, fixed=1),
        P('neuron_type',     ntype, fixed=1),
        P('model',           modeltype,     fixed=1))

    fitness = aju.fitnesses.combined_fitness('empty',
                                             response=2,
                                             baseline_pre=1,
                                             baseline_post=1,
                                             rectification=2,
                                             falling_curve_time=1,
                                             spike_time=1,
                                             spike_width=2,
                                             spike_height=2,
                                             spike_latency=0,
                                             spike_count=1,
                                             spike_ahp=2.5,
                                             ahp_curve=2,
                                             spike_range_y_histogram=1,
                                             mean_isi=2,
                                             isi_spread=1)

    return params,fitness

