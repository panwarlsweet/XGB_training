# Conversion XGBoost model to TMVA weight file
python convert_pkl2xml.py --infile test_file.pkl  <br /> 

# Read TMVA weight file
root -l -q Reader_xml.C <br />
