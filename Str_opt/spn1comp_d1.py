#!/usr/bin/env python3

if __name__ == '__main__':

    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:

            def map_func(f,arglist, callback_function= None):
                result = [executor.submit(f,args) for args in arglist]
                # def res_callback(result, callback_function):
                #     for r in result:
                #         result.wait()
                #         callback_function()
                # result[-1].add_done_callback(result,callback_function)
                return result

            import ajustador as aju
            from ajustador.helpers import save_params,converge
            import os
            import A2Acre
            import params_fitness_chan0 as pf
            import fit_commands as fc

            ########### Optimization of EP neurons ##############3
            modeltype='d1d2'
            rootdir=os.getcwd()+'/output/'
            #use 1 and 3 for testing, 200 and 8 for optimization
            generations=300
            popsiz=8
            seed=84362
            #after generations, do 25 more at a time and test for convergence
            test_size=25
            
            ################## neuron /data specific specifications #############
            ntype='D1'
            morph_file='MScell-soma.p'
            dataname='non05Jan2015_SLH004'
            exp_to_fit=A2Acre.alldata[dataname][[1, 5, 14, 17, 22]]
            ghkkluge=1
            savename=dataname+'_'+str(popsiz)+'_'+str(seed)
            dirname='cmaes_'+dataname+'_'+str(seed)+'_'+str(popsiz)
            if not dirname in os.listdir(rootdir):
                os.mkdir(rootdir+dirname) #this form will make the rootdir/output direcotyr, but not the output/tmp directory
            os.chdir(rootdir+dirname)
            
            ######## set up parameters and fitness to be used for all opts  ############
            params1,fitness=pf.params_fitness(morph_file,ntype,modeltype,ghkkluge)

            # set-up and do the optimization 
            fit=fc.fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params1,generations,popsiz, seed, test_size, map_func = None)
            if test_size>0:
                mean_dict,std_dict,CV=converge.iterate_fit(fit,test_size,popsiz,std_crit=0.01,slope_crit=1e-3,max_evals=100000)

            ###########look at results
            #from ajustador import drawing
            #drawing.plot_history(fit1, fit1.measurement)

            startgood=0  #set to 0 to print all
            threshold=10  #set to large number to print all
            save_params.save_params(fit1, startgood, threshold)
            #save_params.persist(fit1,'.')
