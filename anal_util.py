# Author Avrama Blackwell
# Author Emily Wang
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

def load_npz(fname, tile,printstuff=0):
    dat = np.load(fname) #if this stops working, add allow_pickle=True
    
    param= pd.DataFrame(list(dat['params']), columns=list(dat['paramnames']))
    if printstuff:
        print(param.columns, "FEATURES", dat['features'], len(dat['features'])) #names of columns 
    #param.describe() #summary statistics
    
    #Find column with 'total:', anything beyond that is not a feature
    total_loc= [i for i,feat in enumerate(dat['features']) if feat.startswith('total') ][0]
    features = [feat.split('=')[0] for feat in dat['features'][0:total_loc]]
    #treat last feature (total) differently because has neither = nor 'fitness'
    
    features.append(dat['features'][total_loc].split(':')[0])

    model_loc=[i for i, feat in enumerate(dat['features']) if feat.startswith('model')][0]
    neurontype_loc=[i for i, feat in enumerate(dat['features']) if feat.startswith('neuron')][0]

    model= dat['features'][model_loc].split('=')[-1]
    neuron = dat['features'][neurontype_loc].split('=')[-1]

    #print("FITVALUES")
    #print(dat["fitvals"], "LENGTH", len(dat["fitvals"]), len(dat["fitvals"][0]))
    
    fit=pd.DataFrame(dat['fitvals'], columns=features)
    
    #fit.columns #names of columns
    #fit.describe() #summary statistics

    fit_param=pd.concat([param,fit],axis=1) #same number of rows in param and fit, put them "side by side"

    thresh=fit_param.quantile(tile)['total'] #values of 10th percentile
    goodsamples=fit_param[fit_param['total']<thresh]
    goodsamples['neuron'] = neuron
    goodsamples['model'] = model
    
    print('loadnpz:', fname, model, neuron,goodsamples.shape)
    if printstuff:
        print('cell', fname, 'percentile', tile, 'threshold', thresh, 'samples',len(goodsamples),'from',len(fit_param))
    return goodsamples,features

############## 
def calculate_pvalues_2file(df1,df2):
    #df1 is rows, df2 is columns
    pvalues = pd.DataFrame(index=df1.columns, columns=df2.columns)
    corr = pd.DataFrame(index=df1.columns, columns=df2.columns)
    for r in df1.columns:
        for c in df2.columns:
            if not (r == 'cell' or c == 'cell'):
                corr.loc[r,c] = round(pearsonr(df1[r], df2[c])[0], 4)
                pvalues.loc[r,c] = pearsonr(df1[r], df2[c])[1]
    return corr, pvalues

def calculate_pvalues(df1):
    df1_clean = df1.dropna()
    pvalues = pd.DataFrame(columns=df1_clean.columns,index=df1_clean.columns)
    corr=pd.DataFrame(columns=df1_clean.columns,index=df1_clean.columns)
    for i,r in enumerate(df1_clean.columns):
        for c in df1_clean.columns[i+1:]:
            if not (r == 'cell' or c == 'cell'):
                corr.loc[r,c] = round(pearsonr(df1_clean[r], df1_clean[c])[0], 4)
                pvalues.loc[r,c] = pearsonr(df1_clean[r], df1_clean[c])[1]
    return corr, pvalues

#extract only thoses correlations greater than some value (0.5) with sig> value
def create_sig_list(pvalues,corr,sigthresh,corrthresh):
    sig_list=[]
    for i in pvalues.index:
        for j in pvalues.columns:
            if (i != j and pvalues.loc[i,j] < sigthresh and corr.loc[i,j]*corr.loc[i,j]>corrthresh):
                sig_list.append([i,j,corr.loc[i,j]])
    return sig_list

#extract list of variables for scatter plot
def create_var_list(sig_list):
    varlist=[]
    for item in sig_list:
        varlist.append(item[0])
        varlist.append(item[1])
    return np.unique(varlist)

#combined cells into single df
def combined_df(fnames, tile, neurtype,printstuff=0):
    df_list = []
    columnList = []
    #print('fnames:', fnames, 'tile:', tile)
    for fn in fnames:
        good, features = load_npz(fn,tile,printstuff)
        cellname = os.path.basename(fn).split('.')[0]
        print('  *** au.combined_df: fname=',fn,'cellname=',cellname)
        good['cell'] = cellname
        df_list.append(good)
    
        #If a set of optimizations used different features or parameters, should not combine them in classifier; BUT
    #if you insist, this will extract only those feature and params common to the entire set
    columnList = []
    #print('df_list=', df_list)
    for i,df in enumerate(df_list):
        columnList.append(set(df.columns))
        print('  *** columns in df for file ',i,len(df.columns),df['cell'].iloc[0])
    if printstuff:
        print('columnList',columnList)
    commonCol=list(set(x for cols in columnList for x in cols))
    print('number of common columns',len(commonCol))
    
    for df in df_list:
        df = df[commonCol]
    print('after common columns',df.columns,'files',pd.unique(df['cell']))
    
    ## join multiple dataFrames, e.g. join all the proto together    
    alldf=pd.concat(df_list)
    #print('alldf - concatenated', alldf.columns,'files',pd.unique(alldf['cell']))
    
    #print(alldf.columns)
        
    return alldf, df_list


