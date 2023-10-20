#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
''' 
Program to 
a) read in ML model files for a given IPCC region, season and CMIP6 model.
b) Use training data from climate model-a, apply on ML-emulator for climate-model-b (essentially all climate models) and save resulting
 gpp estimates.
c) Calculate Jensen-Shannon distance between gpp estimated by ML emulator for climate model-a and all other climate models.
d) Calculate Jensen-Shannon distance between input vars of climate-model-a and other climate model inputs.
e) Save files for all these J-S distances - for this region/season with input from climate-model-a.
'''


import os, sys, iris
import numpy as np
from optparse import OptionParser

import matplotlib.pyplot as plt
from sklearn import svm, tree
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.inspection import permutation_importance

from joblib import dump, load
import pandas as pd

from numpy import genfromtxt
from scipy.spatial import distance

from file_processing_utils import *

#Global variables
feature_group = 'atmos_3'
predictand_var = 'gpp'
sep_us = '_'
data_dir = '/gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/training/'
param_tuning_dir = '/gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/params/'
model_dir = '/gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/models/'
model_names = ['IPSL-CM6A-LR','UKESM1-0-LL','CanESM5','GISS-E2-1-G','CNRM-ESM2-1']
predictand_var = 'gpp'
method = 'AdaBoost'
jsd_input_dir = '/gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/jsdinputs/' #input for ML models and js distances
jsd_output_dir = '/gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/jsdoutputs/' #output of js distances
rng_seed = 3

#Feature groups
var_names_dict = dict(
    orig_7 = ['pr','tas','tasmax','tasmin','sm','huss','rsds'], #original 7
    base_4 = ['pr','tas','hurs','rsds'], #reduced 4 as base or key vars
    atmos_3 = ['pr','tas','rsds'] #most relevant atmos only vars 
)        


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
        fileparts = [model, 'predictors', season, region,feature_group]
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

def calculate_jsd_distances_for_model_input_and_predictand(min_num_samples,season,region):
    '''
    Calculate Jensen-Shannon distances between
    (a) between input vars and gpp estimates of every pair of climate models
    '''
    num_predictors = len(var_names_dict[feature_group])
    rng = np.random.default_rng(rng_seed)

    for model  in model_names:
        #read in predictand and predictor data
        print(model)
        #read in predictand and predictor data
        fileparts = [model, predictand_var, season, region]
        infile = sep_us.join(fileparts)
        infile = os.path.join(data_dir,infile)
        target_y = np.loadtxt(infile,delimiter=',')
    
        #mask negative gpp values
        mask_val = 9.969209968386869047e+36 #1e20#1e20
        target_y[target_y < 0.0] = mask_val

        
        fileparts = [model, 'predictors', season, region,feature_group]
        infile = sep_us.join(fileparts)
        infile = os.path.join(data_dir,infile)
        print(infile)
        pred_x = np.loadtxt(infile,delimiter = ',')

        #need to remove any rows of data (pred_x or y that has a mask value
        #each row corresponds to 1 time instance and 1 grid cell
        #fix implemented is to combine pred_x and y, remove rows that have mask values
        #later add rows filled with mask values where removed if needed
    
        new_combined_array = np.column_stack((pred_x,target_y))
        modified_array, rows_to_delete = remove_entries_with_mask_vals(new_combined_array)
        print('Max gpp value : ', str(np.nanmax(modified_array[:,num_predictors])))
        [num_samples,num_features] = modified_array.shape
        print(modified_array.shape)
        
        #sample from this to get min samples to keep all tarining data sets from all models at the same size
        rng.shuffle(modified_array)
        new_modified_array = np.copy(modified_array[0:min_num_samples,:])

        #save this array - to reproduce results
                
        #separate the predictors and target data to run through ML models
        pred_x = np.copy(new_modified_array[:,0:num_predictors])
        fileparts = [model, 'min_samples_predictor_vars', season,region,method]
        outfile = sep_us.join(fileparts)
        outfile = os.path.join(jsd_input_dir,feature_group,outfile)
        print(outfile)
        np.savetxt(outfile,pred_x,delimiter = ',')

        target_y = np.copy(new_modified_array[:,num_predictors])
        
        
        fileparts = [model, 'min_samples_predictand_var', season,region,method]
        outfile = sep_us.join(fileparts)
        outfile = os.path.join(jsd_input_dir,feature_group,outfile)
        np.savetxt(outfile,target_y,delimiter = ',')


    # read in input vars for every pair of models and calculate JS-distances
    for model in model_names:
        #read input file
        fileparts = [model, 'min_samples_predictor_vars', season,region,method]
        outfile = sep_us.join(fileparts)
        outfile = os.path.join(jsd_input_dir,feature_group,outfile)
        model1_predictors = np.loadtxt(outfile,delimiter = ',')        
        
        #tas has negative values - remove them so the JSD can be computed
        min_val1 = np.nanmin(model1_predictors[:,1])   
        
        #read input file
        fileparts = [model, 'min_samples_predictand_var', season,region,method]
        outfile = sep_us.join(fileparts)
        outfile = os.path.join(jsd_input_dir,feature_group,outfile)
        model1_predictands = np.loadtxt(outfile)        
        for second_model in model_names:
            #read input file            
            fileparts = [second_model, 'min_samples_predictor_vars', season,region,method]
            outfile = sep_us.join(fileparts)
            outfile = os.path.join(jsd_input_dir,feature_group,outfile)
            model2_predictors = np.loadtxt(outfile,delimiter = ',')        

            #tas has negative values - remove them so the JSD can be computed
            min_val2 = np.nanmin(model2_predictors[:,1])   
            model1_predictors_copy = np.copy(model1_predictors)
            
            if(min_val1 < min_val2):
                model1_predictors_copy[:,1] -= min_val1
                model2_predictors[:,1] -= min_val1
            else:
                model1_predictors_copy[:,1] -= min_val2
                model2_predictors[:,1] -= min_val2

            #calculate distance and save this value
            jsd = distance.jensenshannon(model1_predictors_copy, model2_predictors, axis=1, base = 2.0)
            avg_jsd = np.mean(jsd)
            
            formatted_number = ('{:11.8f}'.format(avg_jsd))
            line_str = str(formatted_number)
            
            
            fileparts = ['input1',model,'input2',second_model,'jsd',season,region,method]
            outfile = sep_us.join(fileparts)
            outfile = os.path.join(jsd_output_dir,feature_group,outfile)
            fh = open(outfile, 'w+')
            fh.write(line_str)
            fh.flush()
            fh.close()

            #read input file            
            fileparts = [second_model, 'min_samples_predictand_var', season,region,method]
            outfile = sep_us.join(fileparts)
            outfile = os.path.join(jsd_input_dir,feature_group,outfile)
            model2_predictands = np.loadtxt(outfile,delimiter = ',')        

            #calculate distance and save this value
            jsd = distance.jensenshannon(model1_predictands, model2_predictands, axis=0, base = 2.0)
            avg_jsd = np.mean(jsd)
            
            formatted_number = ('{:11.8f}'.format(avg_jsd))
            line_str = str(formatted_number)
            
            
            fileparts = ['predictand1',model,'predictand2',second_model,'jsd',season,region,method]
            outfile = sep_us.join(fileparts)
            outfile = os.path.join(jsd_output_dir,feature_group,outfile)
            fh = open(outfile, 'w+')
            fh.write(line_str)
            fh.flush()
            fh.close()


def calculate_jsd_distances_for_models(season,region):
    '''
    Calculate Jensen-Shannon distances between
    (a) input atmos variables of different pairs of climate models.
    (b) between output of gpp estimated where input is from climate-model-a and two distributions
    from emulator for climate-model-a and every other climate-model-b.
    '''
    # read in input vars for every pair of models and calculate JS-distances
    for model in model_names:
        #read input file
        fileparts = [model, 'min_samples_predictor_vars', season,region,method]
        outfile = sep_us.join(fileparts)
        outfile = os.path.join(jsd_input_dir,feature_group,outfile)
        model1_predictors = np.loadtxt(outfile,delimiter = ',')        
        
        #read GPP estimates for input from model estimated with ML emulator of model
        fileparts = ['input', model, 'ml', model,'predictions', season,region,method]
        outfile = sep_us.join(fileparts)
        outfile = os.path.join(jsd_input_dir,feature_group,outfile)
        model1_estimates = np.loadtxt(outfile,delimiter = ',')

        for second_model in model_names:
            #read input file            
            fileparts = [second_model, 'min_samples_predictor_vars', season,region,method]
            outfile = sep_us.join(fileparts)
            outfile = os.path.join(jsd_input_dir,feature_group,outfile)
            model2_predictors = np.loadtxt(outfile,delimiter = ',')        

            #calculate distance and save this value
            jsd = distance.jensenshannon(model1_predictors, model2_predictors, axis=1, base = 2.0)
            avg_jsd = np.mean(jsd)
            
            formatted_number = ('{:11.8f}'.format(avg_jsd))
            line_str = str(formatted_number)
            
            
            fileparts = ['input1',model,'input2',second_model,'jsd',season,region,method]
            outfile = sep_us.join(fileparts)
            outfile = os.path.join(jsd_output_dir,feature_group,outfile)
            fh = open(outfile, 'w+')
            fh.write(line_str)
            fh.flush()
            fh.close()

            
            #read GPP estimates for input from model estimated with ML emulator of second_model
            fileparts = ['input', model, 'ml', second_model,'predictions', season,region,method]
            outfile = sep_us.join(fileparts)
            outfile = os.path.join(jsd_input_dir,feature_group,outfile)
            model2_estimates = np.loadtxt(outfile,delimiter = ',')
            
            #calculate distance and save this value
            
            jsd = distance.jensenshannon(model1_estimates, model2_estimates, axis=0, base = 2.0)
            avg_jsd = np.mean(jsd)
            
            formatted_number = ('{:11.8f}'.format(avg_jsd))
            line_str = str(formatted_number)
            
            fileparts = ['estimates',model,'estimates',second_model,'jsd',season,region,method]
            outfile = sep_us.join(fileparts)
            outfile = os.path.join(jsd_output_dir,feature_group,outfile)
            fh = open(outfile, 'w+')
            fh.write(line_str)
            fh.flush()
            fh.close()
            
    return        
    
    
def run_regional_models_for_min_samples(min_num_samples,season,region):
    ''' 
    For every climate model, take atmos vars as output and use every climate model's 
    ML emulator to estimate GPP. Save predictions as well as inputs for calculating JS distance and 
    reproducability respectively.
    '''
    
    # sample model file : UKESM1-0-LL_AdaBoost_jja_SCA.joblib 
    # Model dir : /gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/models/atmos_3/
    # sample input atmos vars file : UKESM1-0-LL_predictors_jja_WNA_atmos_3
    # sample input gpp file : UKESM1-0-LL_gpp_jja_MED 
    # Training data dir : /gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/training/

    num_predictors = len(var_names_dict[feature_group])
    rng = np.random.default_rng(rng_seed)

    for model  in model_names:
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

        #save this array - to reproduce results
                
        #separate the predictors and target data to run through ML models
        pred_x = np.copy(new_modified_array[:,0:num_predictors])
        fileparts = [model, 'min_samples_predictor_vars', season,region,method]
        outfile = sep_us.join(fileparts)
        outfile = os.path.join(jsd_input_dir,feature_group,outfile)
        np.savetxt(outfile,pred_x,delimiter = ',')

        target_y = np.copy(new_modified_array[:,num_predictors])
        fileparts = [model, 'min_samples_predictand_var', season,region,method]
        outfile = sep_us.join(fileparts)
        outfile = os.path.join(jsd_input_dir,feature_group,outfile)
        np.savetxt(outfile,target_y,delimiter = ',')


        #loop through models and run ML model
        for second_model in model_names:
            #load the ML model file
            fileparts = [second_model,method,season, region]
            outfile = sep_us.join(fileparts)
            outfile = outfile + '.joblib'
            outfile = os.path.join(model_dir,feature_group,outfile)
            regr = load(outfile) 
            y_predicted = regr.predict(pred_x)
        
            #save these predictions as well
            fileparts = ['input', model, 'ml', second_model,'predictions', season,region,method]
            outfile = sep_us.join(fileparts)
            outfile = os.path.join(jsd_input_dir,feature_group,outfile)
            np.savetxt(outfile,y_predicted,delimiter = ',')
            
    return

 
def main():
    
    #Input arguments
    parser = OptionParser()
    parser.add_option("--season", action = "store", type = "string" , dest = "season")
    parser.add_option("--region", action = "store", type = "string" , dest = "region")  
    parser.add_option("--fg", action= "store", type = "string", dest = "feature_group")
    
    
    (options,args) = parser.parse_args()
    
    season = options.season
    feature_group = options.feature_group
    region = options.region

    #read the input data for this climate model and the ML-model files for each other climate model 
    #run the ML models with the input and save gpp estimates
    
    num_predictors = len(var_names_dict[feature_group])
    print("Num predictors", str(num_predictors))

    print('Season: ', season)
    print('Feature group ', feature_group)
    
    min_num_samples = find_least_sample_count(season, region)  
    print('Min num samples ', str(min_num_samples))

    
    # sample model file : UKESM1-0-LL_AdaBoost_jja_SCA.joblib 
    # Model dir : /gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/models/atmos_3/
    # sample input atmos vars file : UKESM1-0-LL_predictors_jja_WNA_atmos_3
    # sample input gpp file : UKESM1-0-LL_gpp_jja_MED 
    # Training data dir : /gws/nopw/j04/ukesm_nceo/ranjinis/gpp-eval-ml/regression/training/
    
   # run_regional_models_for_min_samples(min_num_samples,season,region)
   # calculate_jsd_distances_for_models(season,region)

    #We only want JSD distances for input atmos vars and gpp from climate models - so run just this
    calculate_jsd_distances_for_model_input_and_predictand(min_num_samples,season,region)
    
if __name__ == '__main__':
    main()

