#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
''' 
Program to 
a) read in training data for regression in the form of predictors and gpp files per IPCC AR6 region and for a list of CMIP models,
b) find the minimum number of samples and select that number for regression,
c) split data from each climate model into test and training sets and train across the parameter space
d) Report RMSE values for training and test data across parameter space
e) Pick the best parameter, save model with that param and apply to test data
f) Save predictions, feature importances (permutation variation and RFE)
'''

import os, sys, iris
import numpy as np
from optparse import OptionParser

import matplotlib.pyplot as plt
from sklearn import svm, tgre
from sklearn.metrics import mean_squagd_error
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.inspection import permutation_importance

from joblib import dump, load
import pandas as pd

from numpy import genfromtxt

from file_processing_utils import *

#Global variables
feature_group = 'atmos_3'
predictand_var = 'gpp'
sep_us = '_'
data_dir = '/gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/training/'
param_tuning_dir = '/gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/params/'
model_dir = '/gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/models/'
model_names = ['UKESM1-0-LL','IPSL-CM6A-LR','CanESM5','GISS-E2-1-G','CNRM-ESM2-1']
predictand_var = 'gpp'
method = 'AdaBoost'

#Feature groups
var_names_dict = dict(
    orig_7 = ['pr','tas','tasmax','tasmin','sm','huss','rsds'], #original 7
    base_4 = ['pr','tas','hurs','rsds'], #reduced 4 as base or key vars
    atmos_3 = ['pr','tas','rsds'] #most relevant atmos only vars 
)        
#random seeds
rng_seeds = [12345,5678,94321]
random_seeds = [1,3]
    


DEBUG = 0

def feature_select_rfe(pred_X, target_y, regr, num_features):
    estimator = regr
    selector = RFE(estimator, n_features_to_select= 1, step=1)
    selector = selector.fit(pred_X, target_y)
    return selector.ranking_

def find_least_sample_count(season, region):
    # Read in data for all models and find the count of samples in the smallest data set
    #model_names = ['UKESM1-0-LL','IPSL-CM6A-LR','CanESM5','GISS-E2-1-G','CNRM-ESM2-1']
    #season = 'annual'
    #feature_group = 'orig_7'

    sep_us = '_'

    min_num_samples = -1
    for model in model_names:
        print(model)    
        fileparts = [model, predictand_var, season, region]
        infile = sep_us.join(fileparts)
        infile = os.path.join(data_dir,infile)    
        target_y = np.loadtxt(infile,delimiter=',')
    
   
        #reshape to apply standard scalar -- it should be a single vector
        # target_y = target_y.reshape(-1,1)
        fileparts = [mmodel, 'predictors', season, region,feature_group]
        infile = sep_us.join(fileparts)
        infile = os.path.join(data_dir,infile)
        print(infile)
        pred_x = np.loadtxt(infile,delimiter = ',')
    
        #need to remove any rows of data (pred_x or y that has a mask value
        #each row corresponds to 1 time instance and 1 grid cell
        #fix implemented is to combine pred_x and y, remove rows that have mask values
        #later add rows filled with mask values where removed
    
        new_combined_array = np.column_stack((pred_x,target_y))
        modified_array, rows_to_delete = remove_entries_with_mask_vals(new_combined_array)
    
        #count the number of rows in the modified array returned so you have a count of samples
        [num_samples,num_features] = modified_array.shape
        print(num_samples)
        if(min_num_samples == -1):
            min_num_samples = num_samples
        if(min_num_samples > num_samples):
            min_num_samples = num_samples
    

    return min_num_samples


def do_param_tuning(
    min_num_samples, #Minimum number of training samples across all models considered
    model,  # Model to train on
    season, # Season to train on  
    region,  #Region to train on 
    rng_seed, random_seed): #random ring and random seed for shuffling and running AdaBoost
    
    num_predictors = len(var_names_dict[feature_group])
    rng = np.random.default_rng(rng_seed)

    train_rmse_vals = dict()
    test_rmse_vals = dict()
    depth_vals = [3,10,15,20,25] #train for depth val

    #read in predictand and predictor data
    fileparts = [model, predictand_var, season, region]
    infile = sep_us.join(fileparts)
    infile = os.path.join(data_dir,infile)
    target_y = np.loadtxt(infile,delimiter=',')
    
           
    fileparts = [model, 'predictors', season, region,feature_group]
    infile = sep_us.join(fileparts)
    infile = os.path.join(data_dir,infile)
    print(infile)
    pred_x = np.loadtxt(infile,delimiter = ',')
    train_rmse_vals[model] = np.empty([len(depth_vals)])
    test_rmse_vals[model] = np.empty([len(depth_vals)])

    #need to remove any rows of data (pred_x or y that has a mask value
    #each row corresponds to 1 time instance and 1 grid cell
    #fix implemented is to combine pred_x and y, remove rows that have mask values
    #later add rows filled with mask values where removed if needed
    
    new_combined_array = np.column_stack((pred_x,target_y))
    modified_array, rows_to_delete = remove_entries_with_mask_vals(new_combined_array)
    [num_samples,num_features] = modified_array.shape
    print(modified_array.shape)
           
    #sample from this to get min samples to keep all tarining data sets from all models at the same size
    rng.shuffle(modified_array)
    new_modified_array = modified_array[0:min_num_samples,:]
    print(new_modified_array.shape)
    
    #shuffle again and split the data into take out 15% of the samples to test 
    rng.shuffle(new_modified_array)
    num_test_samples = int(0.15*min_num_samples)
    test_data = new_modified_array[0:num_test_samples,:]
    train_data = new_modified_array[num_test_samples+1:,:]
    
    print(test_data.shape)
    print(train_data.shape)
    
    #files to save the training and test rmse values for this split
    fileparts = [model,method,'rmse', season, region,'train',str(rng_seed),str(random_seed)]
    trainfile = sep_us.join(fileparts)
    trainfile = os.path.join(param_tuning_dir,feature_group,trainfile)
    
    
    fileparts = [model,method,'rmse', season, region,'test',str(rng_seed),str(random_seed)]
    testfile = sep_us.join(fileparts)
    testfile = os.path.join(param_tuning_dir,feature_group,testfile)
    
    train_fh = open(trainfile, 'w+')
    test_fh = open(testfile, 'w+')
   
   
    #file to save test R2 score values for this split
    fileparts = [model,method,'R2_score', season, region,'test',str(rng_seed),str(random_seed)]
    testfile_r2score = sep_us.join(fileparts)
    testfile_r2score = os.path.join(param_tuning_dir,feature_group,testfile_r2score)
    test_r2score_fh = open(testfile_r2score, 'w+') 
   
    #file to save permutation feature importances for this split
    fileparts = [model,method,'feature_perm_scores', season, region,'test',str(rng_seed),str(random_seed)]
    testfile_features_pm = sep_us.join(fileparts)
    testfile_features_pm = os.path.join(param_tuning_dir,feature_group,testfile_features_pm)   
    test_features_pm_fh = open(testfile_features_pm, 'w+')
    
    #file to save ordered features per depth_val
    fileparts = [model, method, 'feature_ranks_rfe', season,region, 'test', str(rng_seed),str(random_seed)]
    testfile_features_rfe = sep_us.join(fileparts)
    testfile_features_rfe = os.path.join(param_tuning_dir,feature_group,testfile_features_rfe)
    testfile_features_rfe = testfile_features_rfe + '.csv'
    test_features_rfe_fh = open(testfile_features_rfe, 'w+')
    
    index = 0
    for depth_val in depth_vals:
        regr = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=depth_val,splitter = 'best',random_state = rng ),n_estimators = 300, random_state=random_seed )
               
        #Training data first
        pred_x = np.copy(train_data[:,0:num_predictors])
        target_y = np.copy(train_data[:,num_predictors])

        regr.fit(pred_x, target_y)                    
              
        #Training RMSE
        y_predicted = regr.predict(pred_x)
        train_rmse_vals[model][index] = np.sqrt(mean_squared_error(target_y, y_predicted))
        print('Train: ',depth_val,' ',train_rmse_vals[model][index])
        formatted_number = ('{:11.8f}'.format(train_rmse_vals[model][index]))
        line_str = str(depth_val) + ',' + str(formatted_number) + '\n'
        train_fh.write(line_str)
        train_fh.flush()

        #Now apply model to unseen test data and repeat
        pred_x = np.copy(test_data[:,0:num_predictors])
        target_y = np.copy(test_data[:,num_predictors])

        #Calculate R2 score to see how much of the variation is capture by the independent features
        r2_score = regr.score(pred_x,target_y)
        formatted_number = ('{:11.8f}'.format(r2_score))
        print('R2 score : ',str(formatted_number))
    
        #save this score so it can be used to calculate best R2 scores across runs
        line_str = str(depth_val) + ',' + str(formatted_number) + '\n'
        test_r2score_fh.write(line_str)
        test_r2score_fh.flush()

        
        result = permutation_importance(regr,pred_x, target_y, n_repeats=50,random_state=random_seed)
        print('######Test#######')
        print(result.importances_mean)
        line_str = str(depth_val) 
        for imps in result.importances_mean:
            formatted_number = ('{:11.8f}'.format(imps))
            line_str = line_str + ',' + str(formatted_number) 

        line_str = line_str + '\n'                    
        test_features_pm_fh.write(line_str)
        test_features_pm_fh.flush()  
                
        print('#############')
        #Get feature ordering using RFE for each depth_val 
        #get the actual feature names for the ranked feature list
        feature_list_orig = var_names_dict[feature_group]
        num_features = len(feature_list_orig)
      
        ranked_features = feature_select_rfe(pred_x,target_y,regr,num_features)
        print(ranked_features)
        ranked_feature_indices = ranked_features - 1  
        sorted_feature_indices = ranked_feature_indices.argsort()
        print(sorted_feature_indices)
            
        actual_ranked_features = [feature_list_orig[i] for i in sorted_feature_indices]
        sep_csv = ','
        features_line = sep_csv.join(actual_ranked_features)
        features_line = str(depth_val) + ',' + features_line + '\n'
       
        print(features_line)
        test_features_rfe_fh.write(features_line)
        test_features_rfe_fh.flush()
        
        y_predicted = regr.predict(pred_x)        
        test_rmse_vals[model][index] = np.sqrt(mean_squared_error(target_y, y_predicted))
        print('Test: ',depth_val,' ',test_rmse_vals[model][index])
        formatted_number = ('{:11.8f}'.format(test_rmse_vals[model][index]))
        line_str = str(depth_val) + ',' + str(formatted_number) + '\n'
        test_fh.write(line_str)
        test_fh.flush()
        index = index + 1 #next depth_val

    train_fh.close()
    test_fh.close()
    test_features_pm_fh.close() 
    test_r2score_fh.close()
    test_features_rfe_fh.close()
    
    if(DEBUG):
       
        train_features_fh.close()
   
    return

def find_best_features(
    model,
    season,
    region,
    depth_val):
    
    df_best_depth = pd.DataFrame(columns = ['depth_val','pr','tas','rsds']) 
    df_r2scores = pd.DataFrame(columns = ['depth_val','r2_score'])
    df_rfe = pd.DataFrame(columns = ['depth_val','first','second','third'])
   
    for rng_seed in rng_seeds:
        for random_seed in random_seeds:

            #Get permutation scores
            fileparts = [model,method,'feature_perm_scores', season, region,'test',str(rng_seed),str(random_seed)]
            infile = sep_us.join(fileparts)
            infile = os.path.join(param_tuning_dir,feature_group,infile)
            df = pd.read_csv(infile)
            df.columns = ['depth_val','pr','tas','rsds']                
            df_row  = pd.DataFrame(df.loc[df['depth_val']== 15])
            df_best_depth = df_best_depth.append(df_row, ignore_index = True)
        
            #get R2 scores
            fileparts = [model,method,'R2_score', season, region,'test',str(rng_seed),str(random_seed)]
            testfile_r2score = sep_us.join(fileparts)
            testfile_r2score = os.path.join(param_tuning_dir,feature_group,testfile_r2score)
            df = pd.read_csv(testfile_r2score)
            df.columns = ['depth_val','r2_score']                
            df_row  = pd.DataFrame(df.loc[df['depth_val']== 15])
            df_r2scores = df_r2scores.append(df_row, ignore_index = True)
        
            #get RFE feature votes
            fileparts = [model,method,'feature_ranks_rfe', season, region,'test',str(rng_seed),str(random_seed)]
            testfile_rfe = sep_us.join(fileparts)
            testfile_rfe = os.path.join(param_tuning_dir,feature_group,testfile_rfe)
            testfile_rfe = testfile_rfe + '.csv'
            df = pd.read_csv(testfile_rfe)
            df.columns = ['depth_val','first','second','third']                
            df_row  = pd.DataFrame(df.loc[df['depth_val']== 15])
            df_rfe = df_rfe.append(df_row, ignore_index = True)

        
    df_r2scores = df_r2scores.drop(['depth_val'],axis = 1)
    df_best_depth = df_best_depth.drop(['depth_val'], axis = 1)  
    df_rfe = df_rfe.drop(['depth_val'], axis = 1)

    #print(df_rfe)
    #print(df_r2scores)

    # calculate average R2 score across splits and then see how much
    # permutation of each feature contributes on average to the average R2 score
    # If a feature(s) contributes more than epsilon or half of the avg R2 score then 
    # pick that feature as being important.

    mean_r2score = df_r2scores.mean(axis = 0)['r2_score']
    epsilon = (mean_r2score/2)
    print('Mean R2 score ', mean_r2score)
    feature_means = df_best_depth.mean(axis = 0)
    best_feature_str = ''
    first_feature = 1
    print(feature_means)

    for name, values in feature_means.iteritems():    
        if (values > epsilon):
            if(first_feature):
                best_feature_str = str(best_feature_str) + str(name)
                first_feature = 0

            else:
                best_feature_str = best_feature_str + '-' + str(name)

    print("Best feature len ",len(best_feature_str))
    if(len(best_feature_str)== 0): #none exceeding threshold
        votes = feature_means.rank(method = 'max',numeric_only = True)
        print('Votes',votes)

        vars = votes.loc[votes == votes.max()] #df.loc[df['first']==df['first'].max()].T
        print(vars)
        best_feature_str = ''
        first = 1
        for name, values in vars.iteritems():  
            print(name)
            if(first):
                best_feature_str = name
                first = 0
            else:
                best_feature_str = best_feature_str + '-' + name
    

    print("PI ",best_feature_str)

      
    #write to output file
    fileparts = [model,method,'best_pi_feature', season, region,'test']
    outfile = sep_us.join(fileparts) + '.txt'
    outfile = os.path.join(param_tuning_dir,feature_group,outfile)
    #outfile = os.path.join('./',outfile)
    with open(outfile, 'w') as f:
        f.write(best_feature_str)
    f.close()
 
    # calculate average best feature across splits from RFE runs
    votes = df_rfe.groupby(['first']).size().rank(method = 'max')
    print(votes)

    vars = votes.loc[votes == votes.max()] #df.loc[df['first']==df['first'].max()].T
    print(vars.shape)
    print(vars)
    best_feature_str = ''
    first = 1
    for name, values in vars.iteritems():    
        if(first):
            best_feature_str = name
            first = 0
        else:
            best_feature_str = best_feature_str + '-' + name
    print("RFE ", best_feature_str)
    #write to output file
    fileparts = [model,method,'best_rfe_feature', season, region,'test']
    outfile = sep_us.join(fileparts) + '.txt'
    outfile = os.path.join(param_tuning_dir,feature_group,outfile)
#    outfile = os.path.join('./',outfile)
    with open(outfile, 'w') as f:
        f.write(best_feature_str)
    f.close()


def create_model_files(model, season, region, depth_val, min_num_samples):
    
    
    num_predictors = len(var_names_dict[feature_group])
    #read gpp and predictor values
    fileparts = [model, predictand_var, season, region]
    infile = sep_us.join(fileparts)
    infile = os.path.join(data_dir,infile)
    
    target_y = np.loadtxt(infile,delimiter=',')
           
    fileparts = [model, 'predictors', season, region,feature_group]
    infile = sep_us.join(fileparts)
    infile = os.path.join(data_dir,infile)
    #print(infile)
    pred_x = np.loadtxt(infile,delimiter = ',')
    
    rng = np.random.default_rng(rng_seeds[0])
    random_seed = random_seeds[0]

    
     #need to remove any rows of data (pred_x or y that has a mask value
    #each row corresponds to 1 time instance and 1 grid cell
    #fix implemented is to combine pred_x and y, remove rows that have mask values
    #later add rows filled with mask values where removed
    
    new_combined_array = np.column_stack((pred_x,target_y))
    modified_array, rows_to_delete = remove_entries_with_mask_vals(new_combined_array)
    #sample from this
    rng.shuffle(modified_array)
    new_modified_array = modified_array[0:min_num_samples,:]
    #print(new_modified_array.shape)
    pred_x = np.copy(new_modified_array[:,0:num_predictors])
    target_y = np.copy(new_modified_array[:,num_predictors])
   
    regr = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=int(depth_val),splitter = 'best',random_state = rng ),n_estimators = 300, random_state=random_seed )
    
    regr.fit(pred_x, target_y)

     #save this model 
    fileparts = [model,method,season, region]
    outfile = sep_us.join(fileparts)
    outfile = outfile + '.joblib'
    outfile = os.path.join(model_dir,feature_group,outfile)
    dump(regr, outfile) 

    

def find_best_model_and_features_deprecated(data_dir, param_tuning_dir, model_dir,predictand_var, feature_group,min_num_samples, method, model, season, region, depth_val, rng_seed, random_seed, rng):

    sep_us = '_'
     #Feature groups
    var_names_dict = dict(
        orig_7 = ['pr','tas','tasmax','tasmin','sm','huss','rsds'], #original 7
        base_4 = ['pr','tas','hurs','rsds'], #reduced 4 as base or key vars
        atmos_3 = ['pr','tas','rsds'] #most relevant atmos only vars 
    )
    num_predictors = len(var_names_dict[feature_group])
   
    num_rows = 1
    num_cols = 1
    print(region)
    row_index = 0
    col_index = 0
    sub_plot_index = 1
    figw, figh = 15,25
    figure, axes = plt.subplots(nrows= num_rows, ncols= num_cols,figsize=(figw,figh))
    figure.tight_layout(h_pad = 5)
    print(model)
    
    #get feature importance with this depth_val        
    fileparts = [model, predictand_var, season, region]
    infile = sep_us.join(fileparts)
    infile = os.path.join(data_dir,infile)
    
    target_y = np.loadtxt(infile,delimiter=',')
           
    fileparts = [model, 'predictors', season, region,feature_group]
    infile = sep_us.join(fileparts)
    infile = os.path.join(data_dir,infile)
    #print(infile)
    pred_x = np.loadtxt(infile,delimiter = ',')
    
    #need to remove any rows of data (pred_x or y that has a mask value
    #each row corresponds to 1 time instance and 1 grid cell
    #fix implemented is to combine pred_x and y, remove rows that have mask values
    #later add rows filled with mask values where removed
    
    new_combined_array = np.column_stack((pred_x,target_y))
    modified_array, rows_to_delete = remove_entries_with_mask_vals(new_combined_array)
    #sample from this
    rng.shuffle(modified_array)
    new_modified_array = modified_array[0:min_num_samples,:]
    #print(new_modified_array.shape)
    pred_x = new_modified_array[:,0:num_predictors]
    target_y = new_modified_array[:,num_predictors]
    #shuffle again and split the data into take out 15% of the samples to test 
    rng.shuffle(new_modified_array)
    num_test_samples = int(0.15*min_num_samples)
    test_data = new_modified_array[0:num_test_samples,:]
    train_data = new_modified_array[num_test_samples+1:,:]
    

    #Do two kinds of feature selection from here - first based on RMSE and second on permutations
        
    #calculate RFE score to look at feature importances wrt accuracy or rmse.
    # We pick the best model from the previous parameter tuning step and train the model with a random 85% of the data
        
    pred_x_train = np.copy(train_data[:,0:num_predictors])
    target_y_train = np.copy(train_data[:,num_predictors])
        
    regr = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=int(depth_val),splitter = 'best',random_state = rng ),n_estimators = 300, random_state=random_seed )
        
        
    regr.fit(pred_x_train, target_y_train)
    
    #save this model 
    fileparts = ['CMIP6',model,method,season, region,feature_group]
    outfile = sep_us.join(fileparts)
    outfile = outfile + '.joblib'
    outfile = os.path.join(model_dir,outfile)
    dump(regr, outfile) 

        
    #get the actual feature names for the ranked feature list
    feature_list_orig = var_names_dict[feature_group]
    num_features = len(feature_list_orig)
      
    ranked_features = feature_select_rfe(pred_x_train,target_y_train,regr,num_features)
    print(ranked_features)
    ranked_feature_indices = ranked_features - 1  
    sorted_feature_indices = ranked_feature_indices.argsort()
    print(sorted_feature_indices)
            
    actual_ranked_features = [feature_list_orig[i] for i in sorted_feature_indices]
    sep_csv = ','
    features_line = sep_csv.join(actual_ranked_features)
    features_line = features_line + '\n'
       
    fileparts = [model, 'feature_rank_rfe', method,season,region,feature_group]
    outfile = sep_us.join(fileparts)
    outfile = os.path.join(param_tuning_dir,outfile)
    outfile = outfile + '.csv'
    print(features_line)
    with open(outfile, 'w') as f:
        f.write(features_line)
    f.close()
        
    #Now apply model to unseen test data and repeat
    pred_x = test_data[:,0:num_predictors]
    target_y = test_data[:,num_predictors]
       
    #Calculate R2 score to see how much of the variation is capture by the independent features
    r2_score = regr.score(pred_x,target_y)
    formatted_number = ('{:11.8f}'.format(r2_score))
    print('R2 score : ',str(formatted_number))
    
    #save this score so it can be used in the plot
    fileparts = ['CMIP6',model, 'R2_score', method,season,region,feature_group]
    outfile = sep_us.join(fileparts)
    outfile = os.path.join(param_tuning_dir,outfile)
    with open(outfile, 'w') as f:
        f.write(formatted_number)
    f.close()
    
    result = permutation_importance(regr,pred_x, target_y, n_repeats=50,random_state=random_seed,scoring = 'r2')
    print(result.importances_mean)
    sorted_importances_idx = result.importances_mean.argsort()
    print(sorted_importances_idx)

       
    #save to features file
    #get the actual feature names for the ranked feature list
    actual_ranked_features = [feature_list_orig[i] for i in sorted_importances_idx[::-1]]
    sep_csv = ','
    features_line = sep_csv.join(actual_ranked_features)
    features_line = features_line + '\n'
    fileparts = [model, 'feature_rank_permutation', method,season,region,feature_group]
    outfile = sep_us.join(fileparts)
    outfile = os.path.join(param_tuning_dir,outfile)
    outfile = outfile + '.csv'
    print(features_line)
    with open(outfile, 'w') as f:
        f.write(features_line)
    f.close()
        
        
    importances = pd.DataFrame(
        result.importances.T,
        columns= feature_list_orig,
    )
    # print(importances)
    
    #save importances and do not plot for now
    fileparts = ['CMIP6',model,method,'features_permutation_boxplot', predictand_var, season, region,'test',str(rng_seed),str(random_seed)]
    testfile_feature_plot = sep_us.join(fileparts)
    testfile_feature_plot = testfile_feature_plot + '.pkl'
    testfile_feature_plot = os.path.join(param_tuning_dir,testfile_feature_plot)
    importances.to_pickle(testfile_feature_plot)
    
    #ax = axes[row_index,col_index]
    #importances.plot.box(ax = axes[row_index,col_index],vert=False, whis=10) 
    #ax.text(0.8, 0.8, str(formatted_number), transform=ax.transAxes)

    #ax.set_title(model)
    #ax.axvline(x=0, color="k", linestyle="--")
    #ax.set_xlabel("Decrease in R2 score")
    # ax.set_ylabel("Features")
    #ax.figure.tight_layout()
    #col_index += 1
    #if(col_index == num_cols):
        #row_index +=1
        #col_index = 0
            
    #fileparts = ['CMIP6',method,'features_permutation_boxplot', predictand_var, season, region,'test',str(rng_seed),str(random_seed)]
    #testfile_feature_plot = sep_us.join(fileparts)
    #testfile_feature_plot = testfile_feature_plot + '.png'
    #testfile_feature_plot = os.path.join(param_tuning_dir,testfile_feature_plot)
    #title_str = "Feature Importances " + " " +region + " " +season
    #figure.suptitle(title_str)
    #plt.savefig(testfile_feature_plot)
   # plt.show()
    
        
def main():
    
    #Input arguments
    parser = OptionParser()
    parser.add_option("--model",action = "store",type = "string", dest = "model")
    parser.add_option("--season", action = "store", type = "string" , dest = "season")
    parser.add_option("--region", action = "store", type = "string" , dest = "region")  
    parser.add_option("--fg", action= "store", type = "string", dest = "feature_group")
    
    
    (options,args) = parser.parse_args()
    
    model = options.model
    season = options.season
    feature_group = options.feature_group
    region = options.region

    num_predictors = len(var_names_dict[feature_group])
    print("Num predictors", str(num_predictors))

    print('Season: ', season)
    print('Feature group ', feature_group)
    
    min_num_samples = find_least_sample_count(season, region)  
    print('Min num samples ', str(min_num_samples))

    for rng_seed in rng_seeds:
        for random_seed in random_seeds:
            #read model data and then sample
            print(model)
            rng = np.random.default_rng(rng_seed)           
            do_param_tuning(min_num_samples,model,season, region,rng_seed, random_seed) #tune for params
    
    # After writing all the RMSE/Feature importance values to file, do the testing
    files = []
 
    for rng_seed in rng_seeds:
        for random_seed in random_seeds:
            fileparts = [model,method,'rmse', season, region,'test',str(rng_seed),str(random_seed)]
            testfile = sep_us.join(fileparts)
            testfile = os.path.join(param_tuning_dir,feature_group,testfile)
            files.append(testfile)
        
    #read RMSE values for different depth levels across n-fold cv files and average to get best depth (param)
    data = genfromtxt(files[0], delimiter=',') 
    for f in files[1:]:
        data += genfromtxt(f, delimiter=',')
    data /= len(files)
      
    [num_levels, avgs] = data.shape
    print(data.shape)
    depth_val = data[0,0]
    
    ###########################################
    #Previously I looked for the inflexion point but with 3 features it seemed best to find the min error
      #  for index in range(0,num_levels):
      #      dy = data[index+1,1]- data[index,1]
      #      if(dy > 0):
      #          depth_val = data[index,0]
      #          break;
    ###########################################
    depth_val_index = np.argmin(data[:,1])
    print(depth_val_index)
    depth_val = data[depth_val_index,0]
    print('Best param ', depth_val)

    print(rng_seed)
    print(random_seed)
    find_best_features(model, season, region, depth_val)
   
    create_model_files(model, season, region, depth_val, min_num_samples)
    
    
    
if __name__ == '__main__':
    main()
                                        
