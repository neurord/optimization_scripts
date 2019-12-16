import ajustador as aju

def params_fitness(morph_file,ntype,modeltype,ghkkluge):

    P = aju.optimize.AjuParam
    params = aju.optimize.ParamSet(
        P('junction_potential', 0, fixed=1),
        P('RA',                 5.3,  min=1,      max=200),
        P('RM',                2.78,   min=0.1,      max=10),
        P('CM',                 0.010, min=0.001,      max=0.03),
        P('Cond_Kir',      9.5,      min=0, max=10),
        P('Eleak', -0.08, min=-0.080, max=-0.030),
        P('Cond_NaF_0',      219e3,      min=0, max=600e3),
        P('Cond_NaF_1',      1878,      min=0, max=10000),
        P('Cond_NaF_2',      878,      min=0, max=10000),
        P('Cond_KaS_0',      599,        min=0, max=2000),
        P('Cond_KaS_1',      372,        min=0, max=2000),
        P('Cond_KaS_2',      37.2,        min=0, max=200),
        P('Cond_KaF_0',      887,        min=0, max=2000),
        P('Cond_KaF_1',      641,        min=0, max=2000),
        P('Cond_KaF_2',      641,        min=0, max=2000),
        P('Cond_Krp_0',      0.05,        min=0, max=60),
        P('Cond_Krp_1',      0.05,        min=0, max=60),
        P('Cond_Krp_2',      0.05,        min=0, max=60),
        P('Cond_SKCa', 1.7, min=0, max=10),
        P('Cond_BKCa', 5.6, min=0, max=50),
        P('Cond_CaN_0',      3*ghkkluge,      min=0, max=100*ghkkluge),
        P('Cond_CaT_1',      2*ghkkluge,      min=0, max=100*ghkkluge),
        P('Cond_CaT_2',      2*ghkkluge,      min=0, max=100*ghkkluge),
        P('Cond_CaL12_0',    8*ghkkluge,      min=0, max=100*ghkkluge),
        P('Cond_CaL12_1',    4*ghkkluge,      min=0, max=100*ghkkluge),
        P('Cond_CaL12_2',    4*ghkkluge,      min=0, max=100*ghkkluge),
        P('Cond_CaL13_0',   12*ghkkluge,      min=0, max=100*ghkkluge),
        P('Cond_CaL13_1',    6*ghkkluge,      min=0, max=100*ghkkluge),
        P('Cond_CaL13_2',    6*ghkkluge,      min=0, max=100*ghkkluge),
        P('Cond_CaR_0',     20*ghkkluge,      min=0, max=1000*ghkkluge),
        P('Cond_CaR_1',     45*ghkkluge,      min=0, max=1000*ghkkluge),
        P('Cond_CaR_2',     45*ghkkluge,      min=0, max=1000*ghkkluge),
        P('morph_file', morph_file, fixed=1),
        P('neuron_type', ntype,                     fixed=1),
        P('model',           modeltype,     fixed=1))

    #fitness=aju.fitnesses.combined_fitness('new_combined_fitness')
    fitness = aju.fitnesses.combined_fitness('empty',
                                             response=1,
                                             baseline_pre=1,
                                             baseline_post=1,
                                             rectification=1,
                                             falling_curve_time=1,
                                             spike_time=0,
                                             spike_width=1,
                                             spike_height=1,
                                             spike_latency=1,
                                             spike_count=1,
                                             spike_ahp=1,
                                             ahp_curve=2,
                                             charging_curve=1,
                                             spike_range_y_histogram=0)
    return params,fitness
