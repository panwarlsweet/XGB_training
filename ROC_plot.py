import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

with open('/afs/cern.ch/user/l/lata/HHbbggTraining/scripts/XGB_training/output_files_nocosHH_nocosbb_nocosgg/lowmass_test_xgbr_diphotonsROC_res.dat', 'r') as f:
    lines = f.readlines()
    x = [float(line.split()[0]) for line in lines]
    y = [float(line.split()[1]) for line in lines]
        
roc_auc = 0.9459   #test_xgbr_gJets
plt.plot(y, x, lw=1, label='ROC (area = %0.4f), $\Delta\eta_{HH}$; no cos*(HH/bb/gg)'%(roc_auc))

with open('/afs/cern.ch/user/l/lata/HHbbggTraining/scripts/XGB_training/output_files_nocos_nodetas/lowmass_test_xgbr_diphotonsROC_res.dat', 'r') as f:
    lines = f.readlines()
    x1 = [float(line.split()[0]) for line in lines]
    y1 = [float(line.split()[1]) for line in lines]
        
roc_auc = 0.9449   #test_xgbr_gJets
plt.plot(y1, x1, lw=1, label='ROC (area = %0.4f), no $\Delta\eta_{HH}$; no cos*(HH/bb/gg)'%(roc_auc))

plt.xlim([-0.05, 0.30])
plt.ylim([0.70, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title("background class = -1")
plt.grid()
plt.savefig("test2.eps")

#plt.show()

