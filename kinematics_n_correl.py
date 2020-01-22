#!/usr/bin/env python
# coding: utf-8

# In[1]:


####### importing module ###########

#get_ipython().run_line_magic('matplotlib', 'inline')
from ROOT import TMVA, TFile, TCut
import random
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, mean_squared_error, auc
from root_numpy import root2array, rec2array
from scipy.stats import uniform, randint
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split, StratifiedKFold
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


### assigning branch from root trees ########

branch_names = ['leadingJet_DeepFlavour','subleadingJet_DeepFlavour','absCosThetaStar_CS','absCosTheta_bb','absCosTheta_gg','diphotonCandidatePtOverdiHiggsM','dijetCandidatePtOverdiHiggsM','leadingJet_bRegNNResolution','subleadingJet_bRegNNResolution','customLeadingPhotonIDMVA','customSubLeadingPhotonIDMVA','PhoJetMinDr','sigmaMJets', 'leadingJet_pt/Mjj', 'subleadingJet_pt/Mjj', 'leadingPhoton_pt/CMS_hgg_mass', 'subleadingPhoton_pt/CMS_hgg_mass', 'rho', 'leadingPhotonSigOverE', 'subleadingPhotonSigOverE', 'sigmaMOverM', 'deltaEtaHH']
branch_names = [c.strip() for c in branch_names]

branch_names = (b.replace(" ", "_") for b in branch_names)

branch_names = list(b.replace("-", "_") for b in branch_names)
print(branch_names)


# In[3]:


#### making array of variables ######

sig = root2array("legacy_branch_flattrees/output_Signal_RD_BG_lowmass.root",
                 "tagsDumper/trees/Signal_13TeV_DoubleHTag_0", 
                 branch_names,
                 selection=''
               )
sig = rec2array(sig)

bkg_dipho = root2array("legacy_branch_flattrees/output_DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa.root",
                 "tagsDumper/trees/Background_13TeV_DoubleHTag_0", 
                 branch_names,
                 selection=''
                )
bkg_dipho = rec2array(bkg_dipho)

bkg_gJet = root2array("legacy_branch_flattrees/output_GJet_20toInf.root",
                 "tagsDumper/trees/Background_13TeV_DoubleHTag_0",
                 branch_names,
                 selection=''
                )

bkg_gJet = rec2array(bkg_gJet)

bkg = [bkg_dipho, bkg_gJet]

# In[4]:

## do it one by one for each bkg element as both array elements in bkg have diff dimension so do it for range(0, 1) and range (1,2)
for b in range(1,2):
        X = np.concatenate((sig, bkg[b]))
        print(X.shape)

        y = np.concatenate((np.ones(sig.shape[0]),
                            np.zeros(bkg[b].shape[0])))   



        # Create a pandas DataFrame for our data
        # this provides many convenience functions
        # for exploring your dataset
        # need to reshape y so it is a 2D array with one column

        from sklearn import preprocessing

        df = pd.DataFrame(np.hstack((X, y.reshape(y.shape[0], -1))),
                          columns=branch_names+['y'])

        data1=df[df.y<0.5]
        data2=df[df.y>0.5]


        from pandas.core.index import Index

        column=["leadingJet_DeepFlavour","subleadingJet_DeepFlavour","absCosThetaStar_CS","absCosTheta_bb", "absCosTheta_gg", "diphotonCandidatePtOverdiHiggsM", "dijetCandidatePtOverdiHiggsM", "leadingJet_bRegNNResolution", "subleadingJet_bRegNNResolution","customLeadingPhotonIDMVA","customSubLeadingPhotonIDMVA",'PhoJetMinDr','sigmaMJets', 'leadingJet_pt/Mjj', 'subleadingJet_pt/Mjj', 'leadingPhoton_pt/CMS_hgg_mass', 'subleadingPhoton_pt/CMS_hgg_mass', 'rho', 'leadingPhotonSigOverE', 'subleadingPhotonSigOverE', 'sigmaMOverM', 'deltaEtaHH']
        print("read the branches")

        if column is not None:
                if not isinstance(column, (list, np.ndarray, Index)):
                        column = [column]
                        data1 = data1[column]
                        data2 = data2[column]

        data1 = data1._get_numeric_data()
        data2 = data2._get_numeric_data()
                        
        naxes = len(data1.columns)
        print("naxes==",naxes)
        fig, axes = plt.subplots(8, 3, figsize=(30, 30))
        bins=20
        _axes = axes.flatten()

        for i, col in enumerate(column):
                ax = _axes[i]
                low = min(data1[col].min(), data2[col].min())
                high = max(data1[col].max(), data2[col].max())
                weights1 = np.ones_like(data1[col])/float(len(data1[col]))
                weights2 = np.ones_like(data2[col])/float(len(data2[col]))
                ax.hist(data1[col].dropna().values,
                        bins=bins, range=(low,high), label='Background',  alpha=0.6, weights=weights1)
                ax.hist(data2[col].dropna().values,
                        bins=bins, range=(low,high), label='Signal', alpha=0.6, weights=weights2)
                ax.set_title(col)
                ax.grid(True)
                ax.legend(loc='upper center')
                                
        print("saving_sig_vs_bkg")

        plt.savefig('sig_vs_bkg'+str(b)+'.png')
        plt.savefig('sig_vs_bkg'+str(b)+'.pdf')
        plt.clf()
        #### code for co-relation matrix ########################################

        bg = df.y < 0.5
        sig = df.y > 0.5
        l = ["bkg", "sig"]
        i=0

        def correlations(data, **kwds):
                # Calculate pairwise correlation between features.
                # 
                # Extra arguments are passed on to DataFrame.corr()

                # simply call df.corr() to get a table of
                # correlation values if you do not need
                # the fancy plotting
                corrmat = data.corr(**kwds)
                
                fig, ax1 = plt.subplots(ncols=1, figsize=(10,10))
                
                opts = {'cmap': plt.get_cmap("RdBu"),
                        'vmin': -1, 'vmax': +1}
                heatmap1 = ax1.pcolor(corrmat, **opts)
                plt.colorbar(heatmap1, ax=ax1)
                
                ax1.set_title("Correlations"+ "_"+l[i])
                labels = corrmat.columns.values
                for ax in (ax1,):
                        # shift location of ticks to center of the bins
                        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
                        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
                        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
                        ax.set_yticklabels(labels, minor=False)
                        
                plt.tight_layout()
        
                # remove the y column from the correlation matrix
                # after using it to select background and signal

        correlations(df[bg].drop('y', 1))
        plt.savefig('bkg_correl'+str(b)+'.png')
        plt.savefig('bkg_correl'+str(b)+'.pdf')
        plt.clf()
        i=1
        correlations(df[sig].drop('y', 1))
        if(b==1): break
        plt.savefig('sig_correl.png')
        plt.savefig('sig_correl.pdf')



