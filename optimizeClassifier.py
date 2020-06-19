
import os
import sys
import training_utils as utils
import numpy as np
from importlib import reload
reload(utils)
import preprocessing_utils as preprocessing
reload(preprocessing)
#import plotting_utils as plotting
#reload(plotting)
import postprocessing_utils as postprocessing
reload(postprocessing)
import time
import datetime
start_time = time.time()

ntuples = 'training_files_with_25GeVjetpt'

#signal_BG = ["output_GluGluToBulkGravitonToHHTo2B2G_M-250_350.root"]
signal_RD = ["output_GluGluToRadionToHHTo2B2G_M-250_350.root"]
diphotonJets = ["output_DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa.root"]
#2016                                                                                          
gJets_lowPt = ["output_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCUETP8M1_13TeV_Pythia8.root"]
gJets_highPt = ["output_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCUETP8M1_13TeV_Pythia8.root"]

#utils.IO.add_signal(ntuples,signal_BG,1)
utils.IO.add_signal(ntuples,signal_RD,1)
utils.IO.add_background(ntuples,diphotonJets,-1)
utils.IO.add_background(ntuples,gJets_lowPt,-1)
utils.IO.add_background(ntuples,gJets_highPt,-1)


for i in range(len(utils.IO.backgroundName)):        
    print ("using background file n."+str(i)+": "+utils.IO.backgroundName[i])
for i in range(len(utils.IO.signalName)):    
    print ("using signal file n."+str(i)+": "+utils.IO.signalName[i])

#use noexpand for root expressions, it needs this file https://github.com/ibab/root_pandas/blob/master/root_pandas/readwrite.py
#standart of input values 
#branch_names = 'absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,PhoJetMinDr,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingJet_DeepCSV,subleadingJet_DeepCSV,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,leadingJet_bRegNNResolution,subleadingJet_bRegNNResolution,noexpand:sigmaMJets/Mjj'.split(",")
#st values with adding pt_gg/m_gg
#branch_names = 'absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,PhoJetMinDr,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingJet_DeepCSV,subleadingJet_DeepCSV,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,leadingJet_bRegNNResolution,subleadingJet_bRegNNResolution,noexpand:sigmaMJets/Mjj,noexpand:leadingPhoton_pt/CMS_hgg_mass,noexpand:subleadingPhoton_pt/CMS_hgg_mass'.split(",")
#st values with adding pt_gg/m_gg, pt_jj/M_jj

branch_names = 'absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,PhoJetMinDr,PhoJetOtherDr,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingJet_DeepFlavour,subleadingJet_DeepFlavour,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,leadingJet_bRegNNResolution,subleadingJet_bRegNNResolution,noexpand:sigmaMJets/Mjj,noexpand:leadingPhoton_pt/CMS_hgg_mass,noexpand:subleadingPhoton_pt/CMS_hgg_mass,noexpand:leadingJet_pt/Mjj,noexpand:subleadingJet_pt/Mjj,rho'.split(",")
extra_branches = ['event','weight','btagReshapeWeight','leadingJet_hflav','leadingJet_pflav','subleadingJet_hflav','subleadingJet_pflav','puweight']
branch_names = [c.strip() for c in branch_names]

print( branch_names)

import pandas as pd
import root_pandas as rpd
from root_numpy import root2array, list_trees

for i in range(len(utils.IO.backgroundName)):        
    print (list_trees(utils.IO.backgroundName[i]))
        
preprocessing.set_signals_and_backgrounds("tagsDumper/trees/bbggtrees_13TeV_DoubleHTag_0",branch_names+extra_branches)
X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.set_variables(branch_names)

#relative weighting between components of one class is kept, all classes normalized to the same
weights_bkg,weights_sig=preprocessing.normalize_process_weights(weights_bkg,y_bkg,weights_sig,y_sig)

X_bkg,y_bkg,weights_bkg = preprocessing.randomize(X_bkg,y_bkg,weights_bkg)
X_sig,y_sig,weights_sig = preprocessing.randomize(X_sig,y_sig,weights_sig)

print (X_bkg.shape)
print (y_bkg.shape)
#bbggTrees have by default signal and CR events, let's be sure that we clean it
X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.clean_signal_events(X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig)
#print X_bkg.shape
#print y_bkg.shape

y_total_train = preprocessing.get_total_training_sample(y_sig,y_bkg).ravel()
X_total_train = preprocessing.get_total_training_sample(X_sig,X_bkg)

y_total_test = preprocessing.get_total_test_sample(y_sig,y_bkg).ravel()
X_total_test = preprocessing.get_total_test_sample(X_sig,X_bkg)

w_total_train = preprocessing.get_total_training_sample(weights_sig,weights_bkg).ravel()
w_total_test = preprocessing.get_total_test_sample(weights_sig,weights_bkg).ravel()

from sklearn.externals import joblib

import xgboost as xgb
#clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
#       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
#       min_child_weight=1e-05, missing=None, n_estimators=1500, nthread=1,#4 the same as in the engine scipt smp 4
#       objective='binary:logistic', reg_alpha=0, reg_lambda=0.1,
#       scale_pos_weight=1, seed=0, silent=True, subsample=1)

clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1e-05, missing=None, n_estimators=1000, nthread=4,#4 the same as in the engine scipt smp 4
       objective='binary:logistic', reg_alpha=0, reg_lambda=0.1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)


import optimization_utils as optimization
reload(optimization)
#param_grid = {
#              'n_estimators': [20,40,50],
#              'learning_rate': [0.1,0.2,1.],    
#              'max_depth': [3,4]
#              }
#all
param_grid = {'n_estimators': [4000, 4500, 5000],
              'max_depth': [5,8,10,12,15],
	      'gamma' : [0],  
              'learning_rate': [0.01],    
              'reg_lambda':[0.3],
	      'reg_alpha':[0.01],
              'min_child_weight':[1e-06]
	     }

#optimization.setupJoblib("ivovtin")

#optimization.optimize_parameters_randomizedCV(clf,X_total_train,y_total_train,param_grid,cvOpt=5,nIter=500,nJobs=8)
#all
#clf = optimization.optimize_parameters_randomizedCV(clf,X_total_train,y_total_train,param_grid,cvOpt=3,nIter=200,nJobs=23)
clf = optimization.optimize_parameters_randomizedCV(clf,X_total_train,y_total_train,param_grid,cvOpt=5,nIter=10,nJobs=4)
#

print ('It took', time.time()-start_time, 'seconds.')


