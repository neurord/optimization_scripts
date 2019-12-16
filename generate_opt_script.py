#automatically generate optimization scripts for different gp cell types
import glob

#fname is the name of an existing (set of) optimization scripts
oldtype='proto'
oldcell='154'
fname='gp_opt/gp_'+oldtype+oldcell+'*opt.py'

#list of cell types and cell names for creating new scripts
#newcells=['144','122','079','033','140','138','120']
#newtypes=['proto','proto','proto','arky','arky','arky','arky']
newcells=['5']
newtypes=['npas']
newseed_increment=[0,22] #create new seed by adding this value to the old seed
#generate a new set of optimization scripts for each new cell
for seed in newseed_increment:
    for newcell,newtype in zip(newcells,newtypes):
        #read in the set of files, e.g. that differ by random seed or population size
        files=glob.glob(fname)
        if newcells.index(newcell)==0:
            if len(files)==0:
                print('no files found, pattern is',fname)
            else:
                print('working from file templates:', files)
        for f in files:
            #open the file and read in the lines
            fold=open(f)
            fdata=fold.readlines()
            #find the lines which specify cell type/name
            datalines=[i for i in range(len(fdata)) if oldtype in fdata[i]]
            #update those lines with the new cell type and name
            for i in datalines:
                if oldcell in fdata[i]:
                    fdata[i]=fdata[i].replace(oldtype,newtype).replace(oldcell,newcell)
                else:
                    fdata[i]=fdata[i].replace(oldtype,newtype)
            d=[i for i in range(len(fdata)) if "seed=" in fdata[i]][0]
            oldseed = int(fdata[d].split('=')[1])
            newseed=oldseed+seed
            fdata[d]=fdata[d].replace(str(oldseed),str(newseed))
            #create new file name and write the data
            nameparts=f.split(oldtype+oldcell)
            newname=nameparts[0]+newtype+newcell+"_"+str(newseed)+nameparts[1]
            print(newname)
            with open(newname,'w') as fnew:
                fnew.writelines(fdata)
