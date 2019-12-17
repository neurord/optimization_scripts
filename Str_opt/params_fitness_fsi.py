import ajustador as aju

def params_fitness(morph_file,ntype,modeltype,ghkkluge):

    P = aju.optimize.AjuParam
    params = aju.optimize.ParamSet(
        P('junction_potential', 0, fixed=1),
        P('RM',                1.0,   min=0.01,      max=10),
        P('CM',                 0.007, min=0.001,      max=0.03),
        P('Eleak', -0.08, min=-0.090, max=-0.030),
        P('Cond_NaF_0',      1149,      min=0, max=60000),
        P('Cond_Kv3132_0',      599,        min=0, max=3000),
        P('Cond_Ka_0',      887,        min=0, max=3000),
        P('Cond_Kv13_0',      1500,        min=0, max=6000),
        P('Chan_NaF_vshift_X', 0.0, min=-0.01, max=0.01),
        P('Chan_NaF_vshift_Y', 0.0, min=-0.01, max=0.01),
        P('Chan_NaF_taumul_X', 1.0, min=0.5, max=2),
        P('Chan_NaF_taumul_Y', 1.0, min=0.5, max=2),
        P('Chan_Kv3132_vshift_X', 0.0, min=-0.02, max=0.01),
        P('Chan_Kv3132_taumul_X', 1.0, min=0.6, max=1.8),
        P('Chan_Kv13_vshift_X', 0.0, min=-0.02, max=0.01),
        P('Chan_Kv13_taumul_X', 1.0, min=0.6, max=1.8),
        P('Chan_Ka_vshift_X', 0.0, min=-0.01, max=0.01),
        P('Chan_Ka_vshift_Y', 0.0, min=-0.01, max=0.01),
        P('Chan_Ka_taumul_X', 1.0, min=0.6, max=1.8),
        P('Chan_Ka_taumul_Y', 1.0, min=0.6, max=1.8),
        P('morph_file', morph_file, fixed=1),
        P('neuron_type', ntype,                     fixed=1),
        P('model',           modeltype,     fixed=1))

    #fitness=aju.fitnesses.combined_fitness('new_combined_fitness')
    fitness = aju.fitnesses.combined_fitness('empty',
                                             response=3,
                                             baseline_pre=2,
                                             baseline_post=2,
                                             rectification=1,
                                             falling_curve_time=1,
                                             spike_time=3,
                                             spike_width=2,
                                             spike_height=4,
                                             spike_latency=2,
                                             spike_count=3,
                                             spike_ahp=1,
                                             ahp_curve=2,
                                             mean_isi=2,
                                             isi_spread=2,
                                             #post_injection_curve_tau=2,
                                             #charging_curve_time=1,
                                             charging_curve_full=2,
                                             spike_range_y_histogram=1)
    return params,fitness
