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
            import gpedata_experimental as gpe
            import os
            import fit_commands
            import params_fitness_chan as pf

            ########### Optimization of GP neurons ##############3
            modeltype='gp'
            rootdir=os.getcwd()+'/output/'
            #use 1 and 3 for testing, 200 and 8 for optimization
            generations=1#300
            popsiz=3
            seed=84362
            #after generations, do 25 more at a time and test for convergence
            test_size=0#10#25
            
            ############## neuron /data specific specifications ###########
            ntype='Npas'
            morph_file='GP1_41comp.p'
            dataname='Npas2003'
            exp_to_fit = gpe.data[dataname+'-2s'][[0,2,4]]
            savename=dataname+'_'+str(popsiz)+'_'+str(seed)
            dirname='cmaes_'+dataname+'_'+str(seed)+'_'+str(popsiz)
            if not dirname in os.listdir(rootdir):
                os.mkdir(rootdir+dirname) #this form will make the rootdir/output direcotyr, but not the output/tmp directory
            os.chdir(rootdir+dirname)
            ######## set up parameters and fitness to be used for all opts ####
            params1,fitness=pf.params_fitness(morph_file,ntype,modeltype)

            # set-up and do the optimization 
            fit1,mean_dict1,std_dict1,CV1=fit_commands.fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params1,generations,popsiz, seed, test_size, map_func = map_func)

            ###########look at results
            #drawing.plot_history(fit1, fit1.measurement)

            #Save parameters of good results toward the end, and all fitness values
            startgood=0  #set to 0 to print/save all
            threshold=10  #set to high value to print/save all
            save_params.save_params(fit1, startgood, threshold)
            save_params.persist(fit1,'.')


'''
To create zip file for NSG simulations:
1. create NSG directory
2. place in that directory:
a. ajustador (2nd level)
b. dill
c. igor
d. moose_nerp (2nd level)
e. directory of traces from waves directory, e.g. gepdata_experimental
f. python file that reads in traces, e.g. gpedata-experimental.py
g. optimiztion scripts.py, e.g. gpNpas2003_opt.py
h. create directory output/
2. in parent of NSG directory, zip -r NSGopt.zip NSG/
'''
