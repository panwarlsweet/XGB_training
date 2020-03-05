# XGB_training

In order to perform the training first we need to optimize hyperparameters which could be done using the following command:
````
mkdir output_files
python optimizeClassifier.py
`````
Make sure you specify the input signal and background files' path with all input training variables properly in optimizeClassifier.py. 
Also note that by default the splitting between training and testing is 80/20 if you want to change then correct it here 
https://github.com/panwarlsweet/XGB_training/blob/master/preprocessing_utils.py#L102-L110

Once it gets completed you should be able to get the result in the form of text files in output_files directory. 
Now paste the best esimator from text file here 
https://github.com/panwarlsweet/XGB_training/blob/master/trainMVAHHbbgg_detaHH.py#L78-L83
and do
```````
python trainMVAHHbbgg_detaHH.py 
```````

At the end you should be able to get all the plots with .pkl file in the same output_files folder.

In order to convert .pkl in the .xml format, you can follow the instructions from readme file in Conversion folder.
