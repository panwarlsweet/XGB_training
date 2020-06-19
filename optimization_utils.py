from sklearn import model_selection
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import training_utils as utils
import optimization_utils as plotting
import time
import datetime
import os
start_time = time.time()       
 
def optimize_parameters_gridCV(classifier,X_total_train,y_total_train,param_grid,cvOpt=3,nJobs=10):
    print ("=====Optimization with grid search cv=====")
    scores = model_selection.cross_val_score(classifier,
                                      X_total_train, y_total_train,
                                      scoring="roc_auc",
                                      n_jobs=nJobs,
                                      cv=cvOpt)
    print ("-Initial Accuracy-")
    print ("Accuracy: %0.5f (+/- %0.5f)"%(scores.mean(), scores.std()))

    
    
    X_train, X_test, y_train, y_test = train_test_split(X_total_train, y_total_train)
    
    clf = model_selection.GridSearchCV(classifier,
                                   param_grid,
                                   cv=cvOpt,
                                   scoring='roc_auc',
                                   n_jobs=nJobs, verbose=1)
    clf.fit(X_train, y_train)
    
    print ("Best parameter set found on development set:")
    print()
    print (clf.best_estimator_)
    print()
    print ("Grid scores on a subset of the development set:")
    print()
    df = pd.DataFrame(clf.cv_results_)[['params', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']]
    print (df)
    print ('the best_params : ', clf.best_params_)
    print ('the best_score  : ', clf.best_score_)
    for k,v in clf.best_params_.items():
        param_grid[k] = v
        
    df.to_csv(utils.IO.plotFolder+'xgb-RandomizedSearchCV-results-01.csv', index=False)                
    
    return clf.cv_results_



def optimize_parameters_randomizedCV(classifier,X_total_train,y_total_train,param_grid,nIter=10,cvOpt=3,nJobs=10,weights=None):
    print ("=====Optimization with randomized search cv=====")
    scores = model_selection.cross_val_score(classifier,
                                      X_total_train, y_total_train,
                                      scoring="roc_auc",
                                      n_jobs=nJobs,
                                      cv=cvOpt)
    print ("-Initial Accuracy-")
    print ("Accuracy: %0.5f (+/- %0.5f)"%(scores.mean(), scores.std()))

    
    X_train, X_test, y_train, y_test = train_test_split(X_total_train, y_total_train)
    
    if weights == None:
        clf = model_selection.RandomizedSearchCV(classifier,
                                   param_grid,
                                   n_iter=nIter,
                                   cv=cvOpt,
                                   scoring='roc_auc',
                                   n_jobs=nJobs, verbose=3)
    else:
        clf = model_selection.RandomizedSearchCV(classifier,
                                   param_grid,
                                   n_iter=nIter,
                                   cv=cvOpt,
                                   scoring='roc_auc',
                                   n_jobs=nJobs, verbose=1,
                                   fit_params={'sample_weight': weights})
    clf.fit(X_train, y_train)
    
    print ("Best parameter set found on development set:")
    print()
    print (clf.best_estimator_)
    print()
    print ("Grid scores on a subset of the development set:")
    print()
#    for params, mean_score, scores in clf.grid_scores_:
#    for params, mean_score, scores in clf.cv_results_:
#        print "%0.4f (+/-%0.04f) for %r"%(mean_score, scores.std(), params)
    df = pd.DataFrame(clf.cv_results_)[['params', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']]
    print (df)
    print ('the best_params : ', clf.best_params_)
    print ('the best_score  : ', clf.best_score_)
    for k,v in clf.best_params_.items():
        param_grid[k] = v
 
    now = str(datetime.datetime.now()).split(' ')[0]
    outstr = "%s_optimization_job"%now
    outputFolder = '/Users/latapanwar/Downloads/MyPhD_project_Res_HH/MVAtraining/preliminary_plots/XGB_training/output_files/'
#    df.to_csv(outputFolder+'RandomizedSearchCV-best_parameters.txt', index=False)                
   
    writeInFile=outputFolder+'best_parameters_%s.txt'%outstr
    if writeInFile!=None:
        print (writeInFile)
#        outFile = open(writeInFile,"w")
        outFile = open(writeInFile,"a+")
        outFile.write("Best parameter set found on development set:\n")
        """
    	outFile.write("%s\n"%clf.best_estimator_)
        """
        outFile.write("Grid scores on a subset of the development set:\n")   
        outFile.write("%s\n"%(df)) 
        outFile.write("the best_params:\n")
        outFile.write("%s\n"%(clf.best_params_))  
        outFile.write("the best_score:\n")
        outFile.write("%s\n"%(clf.best_score_))   
        outFile.write("Time in seconds:\n")
        outFile.write("%s\n"%(time.time()-start_time))   
        outFile.close()   
   
    writeInFile2=outputFolder+'results_%s.txt'%outstr
    if writeInFile2!=None:
        print (writeInFile2)
#        outFile2 = open(writeInFile2,"w")
        outFile2 = open(writeInFile2,"a+")
        outFile2.write("%s\n"%(clf.best_score_))
        outFile2.close()

    return clf.cv_results_
