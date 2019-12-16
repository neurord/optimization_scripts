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
            modeltype='FSI'
            rootdir=os.getcwd()+'/output/'
            #use 1 and 3 for testing, 200 and 8 for optimization
            generations=1#300
            popsiz=3
            seed=84362
            #after generations, do 25 more at a time and test for convergence
            test_size=0#25
            
            ################## neuron /data specific specifications #############
            ntype='FSI'
            morph_file='FSIcell-soma.p'
            dataname=''
            exp_to_fit=A2Acre.alldata[dataname][[1, 5, 12, 15, 19]]
            ghkkluge=1
            savename=dataname+'_'+str(popsiz)+'_'+str(seed)
            dirname='cmaes_'+dataname+'_'+str(seed)+'_'+str(popsiz)
            if not dirname in os.listdir(rootdir):
                os.mkdir(rootdir+dirname) #this form will make the rootdir/output direcotyr, but not the output/tmp directory
            os.chdir(rootdir+dirname)
            
            ######## set up parameters and fitness to be used for all opts  ############
            params1,fitness=pf.params_fitness(morph_file,ntype,modeltype,ghkkluge)

            # set-up and do the optimization 
            fit1,mean_dict1,std_dict1,CV1=fc.fit_commands(dirname,exp_to_fit,modeltype,ntype,fitness,params1,generations,popsiz, seed, test_size, map_func = None)

            ###########look at results
            #from ajustador import drawing
            #drawing.plot_history(fit1, fit1.measurement)

            startgood=0  #set to 0 to print all
            threshold=10  #set to large number to print all
            save_params.save_params(fit1, startgood, threshold)
            #save_params.persist(fit1,'.')
