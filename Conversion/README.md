# Conversion XGBoost model to TMVA weight file
python convert_pkl2xml.py --infile /path/to/file.pkl  <br /> 

# Read TMVA weight file
root -l -q Reader_xml.C'("Signal","Mass_range","Year")' <br />
for ex.:
root -l Reader_xml.C'("Radion", "low", "2016") <br />

NOTE: My setup reads everything from eos and store output at eos
