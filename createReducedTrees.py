import os
import sys; sys.path.append("/afs/cern.ch/user/l/lata/HHbbggTraining/scripts/XGB_training") # to load packages
import training_utils as utils
import numpy as np
import root_pandas as rpd
reload(utils)
import preprocessing_utils as preprocessing
reload(preprocessing)
import plotting_utils as plotting
reload(plotting)
import optimization_utils as optimization
reload(optimization)
import postprocessing_utils as postprocessing
reload(postprocessing)

reload(utils)
reload(preprocessing)
reload(plotting)
reload(optimization)
reload(postprocessing)
###### all this should be given as an argument for file to read input #####
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

if year=="2018" :
        ttH_sim = "output_ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root"
else:
        ttH_sim = "output_ttHToGG_M125_13TeV_powheg_pythia8.root"

ntuples = str(year)
signal = ["output_GluGluTo"+str(sig)+"ToHHTo2B2G_M-"+str(mass_range)+"mass.root"]
diphotonJets = ["output_DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa.root"]
#2016
gJets_lowPt = ["output_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_Tune"+str(tune)+"_13TeV_Pythia8.root"]
gJets_highPt = ["output_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_Tune"+str(tune)+"_13TeV_Pythia8.root"]

ggH = ["output_GluGluHToGG_M-125_13TeV_powheg_pythia8.root"]
vbf = ["output_VBFHToGG_M-125_13TeV_powheg_pythia8.root"]
VH = ["output_VHToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root"]
bbH = ["output_bbHToGG_M-125_4FS_yb2_13TeV_amcatnlo.root"]
ttH = [ttH_sim]
Data= ["Data.root"]

utils.IO.add_signal(ntuples,signal,1)
utils.IO.add_background(ntuples,diphotonJets,-1)
utils.IO.add_background(ntuples,gJets_lowPt,-2)
utils.IO.add_background(ntuples,gJets_highPt,-2)
utils.IO.add_background(ntuples,ggH,-3)
utils.IO.add_background(ntuples,vbf,-4)
utils.IO.add_background(ntuples,VH,-5)
utils.IO.add_background(ntuples,bbH,-6)
utils.IO.add_background(ntuples,ttH,-7)

nBkg = len(utils.IO.backgroundName)
print nBkg

utils.IO.add_data(ntuples,Data,-10)

for i in range(len(utils.IO.backgroundName)):
    print "using background file n."+str(i)+": "+utils.IO.backgroundName[i]


for i in range(len(utils.IO.signalName)):    
    print "using signal file n."+str(i)+": "+utils.IO.signalName[i]

print "using data file: "+ utils.IO.dataName[0]

branch_names = 'absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,PhoJetMinDr,PhoJetOtherDr,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingJet_DeepFlavour,subleadingJet_DeepFlavour,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),noexpand:leadingPhoton_pt/CMS_hgg_mass,noexpand:subleadingPhoton_pt/CMS_hgg_mass,noexpand:leadingJet_pt/Mjj,noexpand:subleadingJet_pt/Mjj,rho,year'.split(",")
extra_branches = ['event','weight','btagReshapeWeight','leadingJet_hflav','leadingJet_pflav','subleadingJet_hflav','subleadingJet_pflav','lumi_fb']

branch_names = [c.strip() for c in branch_names]
print branch_names

import pandas as pd  

import root_pandas as rpd

# now need to shuffle here, we just count events
preprocessing.set_signals_and_backgrounds("tagsDumper/trees/bbggtrees_13TeV_DoubleHTag_0",branch_names+extra_branches,shuffle=False)
X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.set_variables(branch_names)

X_data,y_data,weights_data = preprocessing.set_data("tagsDumper/trees/Data_13TeV_DoubleHTag_0",branch_names)
X_data,y_data,weights_data = preprocessing.clean_signal_events_single_dataset(X_data,y_data,weights_data)

#bbggTrees have by default signal and CR events, let's be sure that we clean it
X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.clean_signal_events(X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig)


# load the model from disk
from sklearn.externals import joblib
###########
##2016
loaded_model = joblib.load(os.path.expanduser(str(pklfolder)+'/'+mass_range+'mass_XGB_training_file.pkl'))

print len(utils.IO.backgroundName)
bkg = []
for i in range(0,len(utils.IO.backgroundName)-1): 
    print "bkg n:"+str(i)
    print X_bkg[y_bkg ==-i-1]
    bkg.append(X_bkg[y_bkg ==-i-1])

#compute the MVA
Y_pred_sig = loaded_model.predict_proba(X_sig)[:,loaded_model.n_classes_-1].astype(np.float64)
print Y_pred_sig 

Y_pred_bkg = []
for i in range(0,len(utils.IO.backgroundName)-1):  
    print i
    Y_pred_bkg.append(loaded_model.predict_proba(bkg[i])[:,loaded_model.n_classes_-1].astype(np.float64))

Y_pred_data = loaded_model.predict_proba(X_data)[:,loaded_model.n_classes_-1].astype(np.float64)
print Y_pred_data 

#Adding additional variables needed
import os
#st + pt/mgg, OR + ptMjj+dR
additionalCut_names = 'MX,Mjj,CMS_hgg_mass'.split(",")

outTag = 'flattrees_L2Regression_resonant_PR1217_PR1220_17Sep2020/WED/flattening_'+mass_range+'mass_'+sig+year+'_L2-regression'

outDir=os.path.expanduser("/eos/user/l/lata/Resonant_bbgg/"+outTag)
if not os.path.exists(outDir):
    os.mkdir(outDir)
    
    
#Save Signal
sig_count_df = rpd.read_root(utils.IO.signalName[0],"tagsDumper/trees/bbggtrees_13TeV_DoubleHTag_0", columns = branch_names+additionalCut_names)
preprocessing.define_process_weight(sig_count_df,utils.IO.sigProc[0],utils.IO.signalName[0])

#nTot is a multidim vector with all additional variables, dictVar is a dictionary associating a name of the variable
#to a position in the vector
nTot,dictVar = postprocessing.stackFeatures(sig_count_df,branch_names+additionalCut_names)
print "Y_pred"
print Y_pred_sig.shape

processPath=os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag+'/'+utils.IO.signalName[0].split("/")[len(utils.IO.signalName[0].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root"
postprocessing.saveTree(processPath,dictVar,nTot,Y_pred_sig)

processPath=os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag+'/'+utils.IO.signalName[0].split("/")[len(utils.IO.signalName[0].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root"
postprocessing.saveTree(processPath,dictVar,nTot,Y_pred_sig,nameTree="reducedTree_sig")
# do
    
# do gJets not in the loop since they have two samples for one process
bkg_1_count_df = rpd.read_root(utils.IO.backgroundName[1],"tagsDumper/trees/bbggtrees_13TeV_DoubleHTag_0", columns = branch_names+additionalCut_names)
preprocessing.define_process_weight(bkg_1_count_df,utils.IO.bkgProc[1],utils.IO.backgroundName[1])

nTot,dictVar = postprocessing.stackFeatures(bkg_1_count_df,branch_names+additionalCut_names)
print nTot.shape

bkg_2_count_df = rpd.read_root(utils.IO.backgroundName[2],"tagsDumper/trees/bbggtrees_13TeV_DoubleHTag_0", columns = branch_names+additionalCut_names)
preprocessing.define_process_weight(bkg_2_count_df,utils.IO.bkgProc[2],utils.IO.backgroundName[2])

nTot_2,dictVar = postprocessing.stackFeatures(bkg_2_count_df,branch_names+additionalCut_names)

nTot_3 = np.concatenate((nTot,nTot_2))
print nTot_3.shape

processPath=(os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag+'/'+utils.IO.backgroundName[1].split("/")[len(utils.IO.backgroundName[1].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root").replace("_Pt-20to40","")
postprocessing.saveTree(processPath,dictVar,nTot_3,Y_pred_bkg[1])

processPath=(os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag+'/'+utils.IO.backgroundName[1].split("/")[len(utils.IO.backgroundName[1].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root").replace("_Pt-20to40","")
postprocessing.saveTree(processPath,dictVar,nTot_3,Y_pred_bkg[1],nameTree="reducedTree_bkg_2")

#Bkgs in the loop - diphotJets and another
for iProcess in range(0,len(utils.IO.backgroundName)):
    ##gJets which are two samples for one process are skipped  !not skipped
    iSample=iProcess
    if iProcess == 1 or iProcess ==2:
            continue
    if iProcess > 2:
        iSample = iProcess-1
    
    print "Processing sample: "+str(iProcess)
    bkg_count_df = rpd.read_root(utils.IO.backgroundName[iProcess],"tagsDumper/trees/bbggtrees_13TeV_DoubleHTag_0", columns = branch_names+additionalCut_names)
    preprocessing.define_process_weight(bkg_count_df,utils.IO.bkgProc[iProcess],utils.IO.backgroundName[iProcess])

    nTot,dictVar = postprocessing.stackFeatures(bkg_count_df,branch_names+additionalCut_names)

    processPath=os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag+'/'+utils.IO.backgroundName[iProcess].split("/")[len(utils.IO.backgroundName[2].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root"
    postprocessing.saveTree(processPath,dictVar,nTot,Y_pred_bkg[iSample])    

    processPath=os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag+'/'+utils.IO.backgroundName[iProcess].split("/")[len(utils.IO.backgroundName[2].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root"
    if "Signal_13TeV_DoubleHTag_0"in processPath:
        treeName = "reducedTree_sig_node_"+str(iProcess-6)
    else:
        treeName = "reducedTree_bkg_"+str(iProcess)
    
    postprocessing.saveTree(processPath,dictVar,nTot,Y_pred_bkg[iSample],nameTree=treeName) 

#save Data                                                                                                                                                                                                                                                     
data_count_df = rpd.read_root(utils.IO.dataName[0],"tagsDumper/trees/Data_13TeV_DoubleHTag_0", columns = branch_names+additionalCut_names)

nTot,dictVar = postprocessing.stackFeatures(data_count_df,branch_names+additionalCut_names,isData=1)

#save preselection data                                                                                                                                                                                                                                        
processPath=os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag+'/'+utils.IO.dataName[0].split("/")[len(utils.IO.dataName[0].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root"
postprocessing.saveTree(processPath,dictVar,nTot,Y_pred_data)

processPath=os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag+'/'+utils.IO.dataName[0].split("/")[len(utils.IO.dataName[0].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root"
postprocessing.saveTree(processPath,dictVar,nTot,Y_pred_data,nameTree="reducedTree_bkg")    

print "cd "+os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag
os.system('hadd '+ os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag+'/'+'Total_preselection_diffNaming.root '+ os.path.expanduser('/eos/user/l/lata/Resonant_bbgg/')+outTag+'/'+'*diffNaming.root')

        



