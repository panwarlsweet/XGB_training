import os
import sys; sys.path.append("/afs/cern.ch/user/l/lata/HHbbggTraining/scripts/XGB_training/python") # to load packages
import training_utils as utils
import numpy as np
from importlib import reload
reload(utils)
import preprocessing_utils as preprocessing
reload(preprocessing)
import optimization_utils as optimization
reload(optimization)
import postprocessing_utils as postprocessing
reload(postprocessing)
from IPython import get_ipython

sig=sys.argv[1]
mass_range=sys.argv[2]
if mass_range=="low":
	mass_point = "250_350"
elif mass_range=="mid":
	mass_point = "400_650"
elif mass_range=="high":
	mass_point = "700_1000"
year=sys.argv[3]
pklfolder=sys.argv[4]
if year=="2016":
        tune = "CUETP8M1"
else:
        tune = "CP5"
ntuples = "Run2_mergedfiles_yearlabel"

signal = ["Run2_"+str(sig)+"_"+str(mass_range)+"mass.root"]
#signal_2016 = ["GluGluTo"+str(sig)+"ToHHTo2B2G_M-"+str(mass_range)+"mass2016"+str(mass_#range)+".root"]
#signal_2017 = ["GluGluTo"+str(sig)+"ToHHTo2B2G_M-"+str(mass_range)+"mass2017"+str(mass_range)+".root"]
#signal_2018 = ["GluGluTo"+str(sig)+"ToHHTo2B2G_M-"+str(mass_range)+"mass2018"+str(mass_range)+".root"]

diphotonJets_2016 = ["DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa2016"+str(mass_range)+".root"]
diphotonJets_2017 = ["DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa2017"+str(mass_range)+".root"]
diphotonJets_2018 = ["DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa2018"+str(mass_range)+".root"]

gJets_lowPt_2016 = ["GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCUETP8M1_13TeV_Pythia82016"+str(mass_range)+".root"]
gJets_highPt_2016 = ["GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCUETP8M1_13TeV_Pythia82016"+str(mass_range)+".root"]

gJets_lowPt_2017 = ["GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia82017"+str(mass_range)+".root"]
gJets_highPt_2017 = ["GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia82017"+str(mass_range)+".root"]

gJets_lowPt_2018 = ["GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia82018"+str(mass_range)+".root"]
gJets_highPt_2018 = ["GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia82018"+str(mass_range)+".root"]
utils.IO.add_signal(ntuples,signal,1)
#utils.IO.add_signal(ntuples,signal_2016,1)
#utils.IO.add_signal(ntuples,signal_2017,1)
#utils.IO.add_signal(ntuples,signal_2018,1)
utils.IO.add_background(ntuples,diphotonJets_2016,-1)
utils.IO.add_background(ntuples,diphotonJets_2017,-1)
utils.IO.add_background(ntuples,diphotonJets_2018,-1)
utils.IO.add_background(ntuples,gJets_lowPt_2016,-2)
utils.IO.add_background(ntuples,gJets_highPt_2016,-2)
utils.IO.add_background(ntuples,gJets_lowPt_2017,-2)
utils.IO.add_background(ntuples,gJets_highPt_2017,-2)
utils.IO.add_background(ntuples,gJets_lowPt_2018,-2)
utils.IO.add_background(ntuples,gJets_highPt_2018,-2)

for i in range(len(utils.IO.backgroundName)):        
    print ("using background file n."+str(i)+": "+utils.IO.backgroundName[i])
for i in range(len(utils.IO.signalName)):    
    print ("using signal file n."+str(i)+": "+utils.IO.signalName[i])


#use noexpand for root expressions, it needs this file https://github.com/ibab/root_pandas/blob/master/root_pandas/readwrite.py
#st values with adding pt_gg/m_gg, pt_jj/M_jj
branch_names = 'absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,PhoJetMinDr,PhoJetOtherDr,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingJet_DeepFlavour,subleadingJet_DeepFlavour,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),noexpand:leadingPhoton_pt/CMS_hgg_mass,noexpand:subleadingPhoton_pt/CMS_hgg_mass,noexpand:leadingJet_pt/Mjj,noexpand:subleadingJet_pt/Mjj,rho,year'.split(",")
extra_branches = ['event','weightXlumi','btagReshapeWeight','leadingJet_hflav','leadingJet_pflav','subleadingJet_hflav','subleadingJet_pflav','puweight']

branch_names = [c.strip() for c in branch_names]
print (branch_names)

import pandas as pd
import root_pandas as rpd
from root_numpy import root2array, list_trees

#for i in range(len(utils.IO.backgroundName)):        
#    print list_trees(utils.IO.backgroundName[i])
        
preprocessing.set_signals_and_backgrounds("bbggtrees",branch_names+extra_branches)
X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.set_variables(branch_names)

#relative weighting between components of one class is kept, all classes normalized to the same
#weights_sig=preprocessing.weight_signal_with_resolution(weights_sig,y_sig)
weights_bkg,weights_sig=preprocessing.normalize_process_weights(weights_bkg,y_bkg,weights_sig,y_sig)

X_bkg,y_bkg,weights_bkg = preprocessing.randomize(X_bkg,y_bkg,weights_bkg)
X_sig,y_sig,weights_sig = preprocessing.randomize(X_sig,y_sig,weights_sig)

print (X_bkg.shape)
print (y_bkg.shape)
#bbggTrees have by default signal and CR events, let's be sure that we clean it
X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.clean_signal_events(X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig)
print (X_bkg.shape)
print (y_bkg.shape)

y_total_train = preprocessing.get_total_training_sample(y_sig,y_bkg).ravel()
X_total_train = preprocessing.get_total_training_sample(X_sig,X_bkg)

y_total_test = preprocessing.get_total_test_sample(y_sig,y_bkg).ravel()
X_total_test = preprocessing.get_total_test_sample(X_sig,X_bkg)

w_total_train = preprocessing.get_total_training_sample(weights_sig,weights_bkg).ravel()
w_total_test = preprocessing.get_total_test_sample(weights_sig,weights_bkg).ravel()

########final optimization with all fixed#######

from sklearn.externals import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error
"""
clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0.0, learning_rate=0.01, max_delta_step=0,
       max_depth=8, min_child_weight=1e-06, missing=None,
       n_estimators=2000, n_jobs=1, nthread=8, objective='binary:logistic',
       random_state=0, reg_alpha=0.01, reg_lambda=0.3, scale_pos_weight=1,
       seed=0, silent=True, subsample=1)
"""

clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.01, max_delta_step=0, max_depth=5,
              min_child_weight=1e-06, missing=None, n_estimators=4000, n_jobs=4,
              nthread=4, objective='binary:logistic', random_state=0,
              reg_alpha=0.01, reg_lambda=0.3, scale_pos_weight=1, seed=0,
              silent=True, subsample=1, verbosity=1)

eval_set = [(X_total_train, y_total_train), (X_total_test, y_total_test)]
clf.fit(X_total_train, y_total_train, sample_weight=w_total_train, eval_set=eval_set, eval_metric=["merror","mlogloss"],early_stopping_rounds=200, verbose=True)
mse = mean_squared_error(y_total_test, clf.predict(X_total_test))
print("MSE: %.4f" % mse)
#clf.evals_result()
print (clf.score(X_total_train,y_total_train))

from xgboost import plot_tree
from sklearn.metrics import accuracy_score
import matplotlib as mpl
mpl.use('Agg')
import plotting_utils as plotting
reload(plotting)
import numpy as np
import matplotlib.pyplot as plt
reload(plt)

outTag = mass_range+'mass'
folder = str(pklfolder)+'_' + str(sig) + '_' + outTag + '_' + str(year)
if not os.path.exists(folder):
    os.mkdir(folder)
joblib.dump(clf, os.path.expanduser(str(folder)+'/'+outTag+'_XGB_training_file.pkl'), compress=9)

#plotting.plot_input_variables(X_sig,X_bkg,branch_names)
#plt.show()
plotting.plot_classifier_output(clf,X_total_train,X_total_test,y_total_train,y_total_test,outString=str(folder)+'/'+outTag+"_classifierOutputPlot_xbrg_test_st_values")
plt.clf()
#plt.show()

#fpr,tpr = plotting.plot_roc_curve(X_total_train,y_total_train,clf)
#plotting.print_roc_report(fpr,tpr)
#plt.savefig(utils.IO.plotFolder+"ROC_train.eps")
#plt.show()
#fpr,tpr = plotting.plot_roc_curve(X_total_test,y_total_test,clf)
#plotting.print_roc_report(fpr,tpr)
#plt.show()

#plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
#plt.savefig(utils.IO.plotFolder+outTag+"importance1.eps")
#plt.show()
#

xgb.plot_importance(clf)
plt.savefig(str(folder)+'/'+outTag+"_importance2.pdf")
plt.clf()
#plt.show()

fpr_dipho_2ndtest_2,tpr_dipho_2ndtest_2 = plotting.plot_roc_curve_multiclass_singleBkg(X_total_test,y_total_test,clf,-1,1,outTag+"_test_xgbr_diphotons",weights=w_total_test)
plotting.print_roc_report(fpr_dipho_2ndtest_2,tpr_dipho_2ndtest_2,outString=str(sig)+"_"+outTag+"_"+str(year)+"_test_xgbr_diphotons")
plt.savefig(str(folder)+'/'+outTag+"_test_xgbr_diphotons.pdf")
plt.clf()
#plt.show()
fpr_gJets_2ndtest_2,tpr_gJets_2ndtest_2 = plotting.plot_roc_curve_multiclass_singleBkg(X_total_test,y_total_test,clf,-2,1,outTag+"_test_xgbr_gJets",weights=w_total_test)
plotting.print_roc_report(fpr_gJets_2ndtest_2,tpr_gJets_2ndtest_2,outString=str(sig)+"_"+outTag+"_"+str(year)+"_test_xgbr_gJets")
plt.savefig(str(folder)+'/'+outTag+"_test_xgbr_gJets.pdf")
plt.clf()
#plt.show()

fpr_dipho_2ndtrain_2,tpr_dipho_2ndtrain_2 = plotting.plot_roc_curve_multiclass_singleBkg(X_total_train,y_total_train,clf,-1,1,outTag+"_train_xgbr_diphotons",weights=w_total_train)
plotting.print_roc_report(fpr_dipho_2ndtrain_2,tpr_dipho_2ndtrain_2,outString=str(sig)+"_"+outTag+"_"+str(year)+"_train_xgbr_diphotons")
plt.savefig(str(folder)+'/'+outTag+"_train_xgbr_diphotons.pdf")
plt.clf() 
#plt.show()
fpr_gJets_2ndtrain_2,tpr_gJets_2ndtrain_2 = plotting.plot_roc_curve_multiclass_singleBkg(X_total_train,y_total_train,clf,-2,1,outTag+"_train_xgbr_gJets",weights=w_total_train)
plotting.print_roc_report(fpr_gJets_2ndtrain_2,tpr_gJets_2ndtrain_2,outString=str(sig)+"_"+outTag+"_"+str(year)+"_train_xgbr_gJets")
plt.savefig(str(folder)+'/'+outTag+"_train_xgbr_gJets.pdf")
plt.clf()
#plt.show()


# #############################################################################
#
# Plot feature importance
#importances = clf.get_fscore()

importances = clf.get_booster().get_score(importance_type='weight')
importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)
importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')
plt.savefig(outTag+"_Importance.pdf")
#
# make predictions for test data
y_pred = clf.predict(X_total_test)
predictions = [round(value) for value in y_pred]    
# evaluate predictions
accuracy = accuracy_score(y_total_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

## retrieve performance metrics
results = clf.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)
## plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.savefig(str(folder)+'/'+outTag+"_XGBoostLogLoss.pdf")

## plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.savefig(str(folder)+'/'+outTag+"_XGBoostClassificationError.pdf")
print("done")



