# Conversion XGBoost model to TMVA weight file
python convert_pkl2xml.py --infile /path/to/file.pkl  <br /> 

# Read TMVA weight file
root -l -q Reader_xml.C'("Signal","Mass_range","Year")' <br />
for ex.:
root -l Reader_xml.C'("Radion", "low", "2016") <br />

NOTE: Setup reads everything from eos; path can be changed here https://github.com/panwarlsweet/XGB_training/blob/master/Conversion/Reader_xml.C#L6 if required, according to input of the commands it reads the flat trees year wise to store MVA

Before running, mention output folder name here https://github.com/panwarlsweet/XGB_training/blob/master/Conversion/Reader_xml.C#L16 (after running it, the trees with MVA output branch will appear here)

Also https://github.com/panwarlsweet/XGB_training/blob/master/Conversion/Reader_xml.C#L17-L22 here it reads MVA xml file and cumulative transformation file, I have arranged it to read these according to mass-range, year and signal name.
