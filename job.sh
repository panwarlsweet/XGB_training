#!/bin/bash
cd /afs/cern.ch/work/l/lata/HHbbgg_analysis/CMSSW_10_2_13/src/
#cmsenv
eval `scramv1 runtime -sh`
echo "CMSSW: "$CMSSW_BASE
#Run your program
cd /afs/cern.ch/work/l/lata/HHbbgg_analysis/XGB_training/XGB_training
#python trainMVAHHbbgg.py Radion low run2 test 
python trainMVAHHbbgg.py $1 $2 $3 $4
