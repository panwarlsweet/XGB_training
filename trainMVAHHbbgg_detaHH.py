import os
import sys; sys.path.append("/afs/cern.ch/user/l/lata/HHbbggTraining/scripts/XGB_training") # to load packages
import training_utils as utils
import numpy as np
reload(utils)
import preprocessing_utils as preprocessing
reload(preprocessing)
import optimization_utils as optimization
reload(optimization)
import postprocessing_utils as postprocessing
reload(postprocessing)
from IPython import get_ipython

ntuples = 'training_files_with_25GeVjetpt'
#signal = ["output_GluGluToBulkGravitonToHHTo2B2G_M-250_350.root"]
signal = ["output_GluGluToRadionToHHTo2B2G_M-700_900.root"]
diphotonJets = ["output_DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa.root"]
#2016
gJets_lowPt = ["output_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCUETP8M1_13TeV_Pythia8.root"]
gJets_highPt = ["output_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCUETP8M1_13TeV_Pythia8.root"]

utils.IO.add_signal(ntuples,signal,1)
utils.IO.add_background(ntuples,diphotonJets,-1)
utils.IO.add_background(ntuples,gJets_lowPt,-2)
utils.IO.add_background(ntuples,gJets_highPt,-2)

for i in range(len(utils.IO.backgroundName)):        
    print "using background file n."+str(i)+": "+utils.IO.backgroundName[i]
for i in range(len(utils.IO.signalName)):    
    print "using signal file n."+str(i)+": "+utils.IO.signalName[i]


#use noexpand for root expressions, it needs this file https://github.com/ibab/root_pandas/blob/master/root_pandas/readwrite.py
#st values with adding pt_gg/m_gg, pt_jj/M_jj
branch_names = 'absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,PhoJetMinDr,PhoJetOtherDr,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingJet_DeepFlavour,subleadingJet_DeepFlavour,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,noexpand:leadingJet_bRegNNResolution*1.4826,noexpand:subleadingJet_bRegNNResolution*1.4826,noexpand:sigmaMJets*1.4826,noexpand:leadingPhoton_pt/CMS_hgg_mass,noexpand:subleadingPhoton_pt/CMS_hgg_mass,noexpand:leadingJet_pt/Mjj,noexpand:subleadingJet_pt/Mjj,rho,deltaEtaHH'.split(",")

branch_names = [c.strip() for c in branch_names]
print branch_names

import pandas as pd
import root_pandas as rpd
from root_numpy import root2array, list_trees

#for i in range(len(utils.IO.backgroundName)):        
#    print list_trees(utils.IO.backgroundName[i])
        
preprocessing.set_signals_and_backgrounds("tagsDumper/trees/bbggtrees_13TeV_DoubleHTag_0",branch_names)
X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.set_variables(branch_names)

#relative weighting between components of one class is kept, all classes normalized to the same
#weights_sig=preprocessing.weight_signal_with_resolution(weights_sig,y_sig)
weights_bkg,weights_sig=preprocessing.normalize_process_weights(weights_bkg,y_bkg,weights_sig,y_sig)

X_bkg,y_bkg,weights_bkg = preprocessing.randomize(X_bkg,y_bkg,weights_bkg)
X_sig,y_sig,weights_sig = preprocessing.randomize(X_sig,y_sig,weights_sig)

print X_bkg.shape
print y_bkg.shape
#bbggTrees have by default signal and CR events, let's be sure that we clean it
X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.clean_signal_events(X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig)
print X_bkg.shape
print y_bkg.shape

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
clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0.0, learning_rate=0.01, max_delta_step=0,
       max_depth=8, min_child_weight=1e-06, missing=None,
       n_estimators=2000, n_jobs=1, nthread=8, objective='binary:logistic',
       random_state=0, reg_alpha=0.01, reg_lambda=0.3, scale_pos_weight=1,
       seed=0, silent=True, subsample=1)
eval_set = [(X_total_train, y_total_train), (X_total_test, y_total_test)]
clf.fit(X_total_train, y_total_train, sample_weight=w_total_train, eval_set=eval_set, eval_metric=["merror","mlogloss"], verbose=True)
mse = mean_squared_error(y_total_test, clf.predict(X_total_test))
print("MSE: %.4f" % mse)
#clf.evals_result()
print clf.score(X_total_train,y_total_train)

from xgboost import plot_tree
from sklearn.metrics import accuracy_score
import matplotlib as mpl
mpl.use('Agg')
import plotting_utils as plotting
reload(plotting)
import numpy as np
import matplotlib.pyplot as plt
reload(plt)

outTag = 'highmass'
#outTag = '2017/dev_legecy_runII_ptmgg_ptmjj_dR/'
joblib.dump(clf, os.path.expanduser(utils.IO.plotFolder+outTag+'_XGB_training_file.pkl'), compress=9)

#plotting.plot_input_variables(X_sig,X_bkg,branch_names)
#plt.show()
plotting.plot_classifier_output(clf,X_total_train,X_total_test,y_total_train,y_total_test,outString=utils.IO.plotFolder+outTag+"_classifierOutputPlot_xbrg_test_st_values")
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
plt.savefig(utils.IO.plotFolder+outTag+"_importance2.pdf")
plt.clf()
#plt.show()

fpr_dipho_2ndtest_2,tpr_dipho_2ndtest_2 = plotting.plot_roc_curve_multiclass_singleBkg(X_total_test,y_total_test,clf,-1,1,outTag+"_test_xgbr_diphotons",weights=w_total_test)
plotting.print_roc_report(fpr_dipho_2ndtest_2,tpr_dipho_2ndtest_2,outString=outTag+"_test_xgbr_diphotons")
plt.savefig(utils.IO.plotFolder+outTag+"_test_xgbr_diphotons.pdf")
plt.clf()
#plt.show()
fpr_gJets_2ndtest_2,tpr_gJets_2ndtest_2 = plotting.plot_roc_curve_multiclass_singleBkg(X_total_test,y_total_test,clf,-2,1,outTag+"_test_xgbr_gJets",weights=w_total_test)
plotting.print_roc_report(fpr_gJets_2ndtest_2,tpr_gJets_2ndtest_2,outString=outTag+"_test_xgbr_gJets")
plt.savefig(utils.IO.plotFolder+outTag+"_test_xgbr_gJets.pdf")
plt.clf()
#plt.show()

fpr_dipho_2ndtrain_2,tpr_dipho_2ndtrain_2 = plotting.plot_roc_curve_multiclass_singleBkg(X_total_train,y_total_train,clf,-1,1,outTag+"_train_xgbr_diphotons",weights=w_total_train)
plotting.print_roc_report(fpr_dipho_2ndtrain_2,tpr_dipho_2ndtrain_2,outString=outTag+"_train_xgbr_diphotons")
plt.savefig(utils.IO.plotFolder+outTag+"_train_xgbr_diphotons.pdf")
plt.clf()
#plt.show()
fpr_gJets_2ndtrain_2,tpr_gJets_2ndtrain_2 = plotting.plot_roc_curve_multiclass_singleBkg(X_total_train,y_total_train,clf,-2,1,outTag+"_train_xgbr_gJets",weights=w_total_train)
plotting.print_roc_report(fpr_gJets_2ndtrain_2,tpr_gJets_2ndtrain_2,outString=outTag+"_train_xgbr_gJets")
plt.savefig(utils.IO.plotFolder+outTag+"_train_xgbr_gJets.pdf")
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
plt.savefig(utils.IO.plotFolder+outTag+"_XGBoostLogLoss.pdf")

## plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.savefig(utils.IO.plotFolder+outTag+"_XGBoostClassificationError.pdf")
print("done")


