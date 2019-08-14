# Author Emily Wang
#!/usr/bin/env python
# coding: utf-8

#import anal_util from ajustador/FrontNeuroinf
import sys
sys.path.insert(0, "/home/emily/mooseAssign/ajustador/FrontNeuroinf")
import anal_util as au  

import numpy as np
import pandas as pd
import glob
#import the random forest classifier method
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection,metrics,tree
from matplotlib import pyplot as plt
import operator
from matplotlib.colors import ListedColormap


# plotting Graph Function

def plotPredictions(max_feat, train_test, predict_dict, neurtypes, feature_order):
    ########## Graph the output using contour graph
    #inputdf contains the value of a subset of features used for classifier, i.e., two different columns from df
    feature_cols = [feat[0] for feat in feature_order]
    inputdf = alldf[feature_cols[0:max_feat]]
    
    plt.ion()
    edgecolors=['k','none']
    feature_axes=[(i,i+1) for i in range(0,max_feat,2)]
    for cols in feature_axes:
        plt.figure()
        for key,col in zip(train_test.keys(),edgecolors):
            predict=predict_dict[key]
            df=train_test[key][0]
            plot_predict=[neurtypes.index(p) for p in predict]
            plt.scatter(df[feature_cols[cols[0]]], df[feature_cols[cols[1]]], c=plot_predict,
                        cmap=ListedColormap(['r', 'b']),
                        edgecolor=col, s=20)
            plt.xlabel(feature_cols[cols[0]])
            plt.ylabel(feature_cols[cols[1]])


# running Random Classifier

def runClusterAnalysis(param_values, labels, num_features, alldf):

    ############ Now, data is ready for the cluster analysis ##################
    #select a random subset of data for training, and use the other part for testing
    #sklearn.model_selection.train_test_split(*arrays, **options)
    #returns the top max_feat number of features and their weights
    df_values_train, df_values_test, df_labels_train, df_labels_test = model_selection.train_test_split(param_values, labels, test_size=0.33)
    train_test = {'train':(df_values_train,df_labels_train), 'test':(df_values_test, df_labels_test)}

    #number of estimators (n_estim) is number of trees in the forest
    #This is NOT the number of clusters to be found
    #max_feat is the number of features to use for classification
    #Empirical good default value is max_features=sqrt(num_features) for classification tasks
    max_feat=int(np.ceil(np.sqrt(num_features)))
    n_estim=10
    rtc = RandomForestClassifier(n_estimators=n_estim, max_features=max_feat)

    #This line actually builds the random forest (does the training)
    rtc.fit(df_values_train, df_labels_train)

    ###### EVALUATE THE RESULT
    #calculate a score, show the confusion matrix
    predict_dict = {}
    for nm,(df,labl) in train_test.items():
        predict = rtc.predict(df)
        predict_dict[nm] = predict

    #evauate the importance of each feature in the classifier
    #The relative rank (i.e. depth) of a feature used as a decision node in a tree can be used to assess the relative importance of that feature with respect to the predictability of the target variable. 
    feature_order = sorted({feature : importance for feature, importance in zip(list(df_values_train.columns), list(rtc.feature_importances_))}.items(), key=operator.itemgetter(1), reverse=True)

    
    ###### 3d, plot amd print the predictions of the actual data -- you can do this if # of epochs is low
    plotPredictions(max_feat, train_test, predict_dict, neurtypes, feature_order)
    print('best features',feature_order[0:max_feat])
    return feature_order[0:max_feat]


# # Setting Up Data Files for Cluster Analysis

def set_up_df(neurtypes,path_root='./',tile=0.005): #take pattern: ex. "/path/fileroot"
    #set of data files from parameter optimization
    pattern = path_root+'*.npz'
    #only return the best %tile of samples, or the best num_fits

    num_fits = 10
    #if small=True, use num_fits from each optimization, else, use %tile
    small = True

    #retrieve data files -- sort the files by which neurtype
    fnames = glob.glob(pattern)
    group_names = {key:[f for f in fnames if key in f] for key in neurtypes}
    
    print(fnames,'found by searching for', pattern)
    print(group_names)
    
    ##### process all examples of each type, combine into dict of data frames and then one dataframe
    df_list = {}
    num_good_samples = {}
    for neur in neurtypes:
        df_list[neur], num_good_samples[neur] = au.combined_df(group_names[neur], tile, neur)
        #df_list[neur] is a DATAFRAME
        #num_good_samples[neur] is a LIST OF DATAFRAMES (1 dataframe per npz file)

    #alldf = pd.concat([df for dfset in df_list.values() for df in dfset])  #if df_list[neur] is a list
    alldf = pd.concat([df for df in df_list.values()])
    print('Neuron_types: ', pd.unique(alldf['neuron']), 'df shape', alldf.shape)
    
    ####create smaller df using just small and same number of good fits from each neuron
    min_samples = np.min([n.shape[0] for vals in num_good_samples.values() for n in vals])
    num_samples = 100#min(min_samples, num_fits)
    smalldf_list = {}
    for neur in neurtypes:
        smalldf_list[neur] = df_list[neur][-num_samples:]       #[df[-num_samples:] for df in df_list[neur]]
    if small:
        alldf = pd.concat(df for df in smalldf_list.values())   #[df for dfset in smalldf_list.values() for df in dfset])
        print('SMALLER SET OF SAMPLES: Neuron_types: ', pd.unique(alldf['neuron']), 'df shape', alldf.shape)

    #exclude entire row (observation) if Nan is found
    print(alldf)
    
    alldf = alldf.dropna(axis=1)
    
    print("DROP ALL NA")
    print(alldf)
    #identify fitness columns and number of features
    fitnesses = [col for col in alldf.columns if 'fitness' in col]
    chan_params = [col for col in alldf.columns if 'Chan' in col]
    num_features = len(alldf.columns)-len(fitnesses)

    print('new shape', alldf.shape,'fitnesses:', len(fitnesses), 'params',num_features, "incorrect number because this does not include model or neuron columns")

    #create dataframe with the 'predictor' parameters - conductance and channel kinetics
    #exclude columns that containing neuron identifier or fitness values
    exclude_columns = fitnesses + ['neuron','neurtype','junction_potential', "model", "cell"] #total? ['neuron','neurtype','junction_potential']
    param_columns = [column for column in list(alldf.columns) if column not in exclude_columns]

    param_values = alldf[param_columns]
    #this is the target values (class labels) of the training data
    labels = alldf['neurtype']
    
    return (param_values, labels, num_features, alldf) 



###### MAIN #######
#parameters to control analysis.  use epochs ~ 10 for debugging
epochs = 1#000
neurtypes = ['arky','proto']  #which neurtypes you are identifying between
path_root='gp_opt/pfc*/' #directory and root file name of set of files
tile=0.005 #do classification on this fraction of fits, to only use the good fits
param_values, labels, num_features, alldf = set_up_df(neurtypes,path_root,tile)

### Calling Cluster Analysis Functions (Adjust Epochs)
# Top 8 features & their weights in each epoch are cumulatively summed in collectionBestFeatures = {feature: totalWeightOverAllEpochs}                                                                                                               
# Top 1 feature in each epoch is stored in collectionTopFeatures = {feature: numberOfTimesAsTopFeatureOverAllEpochs}
collectionBestFeatures = {}
collectionTopFeatures = {}
for epoch in range(0, epochs):
    features = runClusterAnalysis(param_values, labels, num_features, alldf)
    for feat, weight in features:
        if feat not in collectionBestFeatures:          # How is the weight scaled? caution
            collectionBestFeatures[feat] = weight
        else:
            collectionBestFeatures[feat] += weight
            
    f, w = features[0]
    if f not in collectionTopFeatures:
        collectionTopFeatures[f] = 1
    else:
        collectionTopFeatures[f] += 1


#### Plotting collectionBestFeatures

listBestFeatures = sorted(collectionBestFeatures.items(), key=operator.itemgetter(1), reverse=True)

objects = [name for name, weight in listBestFeatures] 
y_pos = np.arange(len(listBestFeatures))
performance = [weight for name, weight in listBestFeatures]
f = plt.figure(figsize=(12,8))

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xticks(rotation=90)
plt.ylabel('Total Weight over 1000 epochs')
plt.title('Feature')

plt.show()

#### Plotting collectionTopFeatures

listBestFeatures = sorted(collectionTopFeatures.items(), key=operator.itemgetter(1), reverse=True)

objects = [name for name, weight in listBestFeatures] 
y_pos = np.arange(len(listBestFeatures))
performance = [weight for name, weight in listBestFeatures]
f = plt.figure(figsize=(12,8))

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xticks(rotation=90)
plt.ylabel('Frequency as Top Feature')
plt.title('Feature')

plt.show()

import collections
sorted_x = sorted(collectionBestFeatures.items(), key=operator.itemgetter(1), reverse=True)
print(*sorted_x, sep = "\n") 


###### NOTES FROM SOMEBODY

########################### NEXT, need to do cluster analysis when labels are not know and best features are not known ##########
### e.g. using the hierarchical clustering in SAS, but need a method better than disciminant analysis to select features ###
# Explains different methods for evaluating clusters:
#https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
# TODO
# How to further simplify tree to comment on the entire forest behaviour.
#        What is the meaning of tree.dot
# each optimization gives different results in terms of important features - how to resolve
# label neurons in scatter plot based on neuron type('proto', 'arky'), and add legend
# use neuron number and random seed to label the different clusters.
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html#sphx-glr-auto-examples-ensemble-plot-forest-iris-py
#
#What about using random forest to select parameters, and then hierarchical using those parameters?

#MAY NEED to evaluate how results vary with max_feat and n_etim
#https://scikit-learn.org/stable/modules/ensemble.html#random-forests



