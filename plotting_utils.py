import training_utils as utils

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import plotting_utils as plotting

# ---------------------------------------------------------------------------------------------------
def plot_classifier_output(clf,X_total_train,X_total_test,y_total_train,y_total_test,outString=None):
    

    sig_train = X_total_train[y_total_train > 0]
    bkg_train = X_total_train[y_total_train < 0]
    sig_test = X_total_test[y_total_test > 0]
    bkg_test = X_total_test[y_total_test < 0]

    #if n_classses > 2 sig proba is the last one (in the way the code is written) 
    Y_pred_sig_train = clf.predict_proba(sig_train)[:,clf.n_classes_-1]
    Y_pred_bkg_train = clf.predict_proba(bkg_train)[:,clf.n_classes_-1]
    Y_pred_sig_test = clf.predict_proba(sig_test)[:,clf.n_classes_-1]
    Y_pred_bkg_test = clf.predict_proba(bkg_test)[:,clf.n_classes_-1]




    weights_sig_train = (np.ones_like(sig_train)/float(len(sig_train)))[:,1]
    weights_bkg_train = (np.ones_like(bkg_train)/float(len(bkg_train)))[:,1]
    weights_sig_test = (np.ones_like(sig_test)/float(len(sig_test)))[:,1]
    weights_bkg_test = (np.ones_like(bkg_test)/float(len(bkg_test)))[:,1]



    # This will be the min/max of our plots
    c_max = max(np.max(d) for d in np.concatenate([Y_pred_sig_train,Y_pred_bkg_train,Y_pred_sig_test,Y_pred_bkg_test]))
    c_min = min(np.min(d) for d in np.concatenate([Y_pred_sig_train,Y_pred_bkg_train,Y_pred_sig_test,Y_pred_bkg_test]))

    # Get histograms of the classifiers
    Histo_training_S = np.histogram(Y_pred_sig_train,bins=40,range=(c_min,c_max),weights=weights_sig_train)
    Histo_training_B = np.histogram(Y_pred_bkg_train,bins=40,range=(c_min,c_max),weights=weights_bkg_train)
    Histo_testing_S = np.histogram(Y_pred_sig_test,bins=40,range=(c_min,c_max),weights=weights_sig_test)
    Histo_testing_B = np.histogram(Y_pred_bkg_test,bins=40,range=(c_min,c_max),weights=weights_bkg_test)


    
    # Lets get the min/max of the Histograms
    AllHistos= [Histo_training_S,Histo_training_B,Histo_testing_S,Histo_testing_B]
    h_max = max([histo[0].max() for histo in AllHistos])*1.2
    h_min = min([histo[0].min() for histo in AllHistos])
    
    # Get the histogram properties (binning, widths, centers)
    bin_edges = Histo_training_S[1]
    bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
    bin_widths = (bin_edges[1:] - bin_edges[:-1])
    
    # To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
    ErrorBar_testing_S = np.sqrt(Histo_testing_S[0]/Y_pred_sig_test.size)
    ErrorBar_testing_B = np.sqrt(Histo_testing_B[0]/Y_pred_bkg_test.size)
    

    # Draw objects
    ax1 = plt.subplot(111)
    
    # Draw solid histograms for the training data
    ax1.bar(bin_centers-bin_widths/2.,Histo_training_S[0],facecolor='blue',linewidth=0,width=bin_widths,label='S (Train)',alpha=0.5)
    ax1.bar(bin_centers-bin_widths/2.,Histo_training_B[0],facecolor='red',linewidth=0,width=bin_widths,label='B (Train)',alpha=0.5)
    #ax1.bar(bin_centers,Histo_training_S[0],facecolor='blue',linewidth=0,width=bin_widths,label='S (Train)',alpha=0.5)
    #ax1.bar(bin_centers,Histo_training_B[0],facecolor='red',linewidth=0,width=bin_widths,label='B (Train)',alpha=0.5)
    
    # # Draw error-bar histograms for the testing data
    ax1.errorbar(bin_centers, Histo_testing_S[0], yerr=ErrorBar_testing_S, xerr=None, ecolor='blue',c='blue',fmt='o',label='S (Test)')
    ax1.errorbar(bin_centers, Histo_testing_B[0], yerr=ErrorBar_testing_B, xerr=None, ecolor='red',c='red',fmt='o',label='B (Test)')
    
    # Make a colorful backdrop to show the clasification regions in red and blue
    ax1.axvspan(0.5, c_max, color='blue',alpha=0.08)
    ax1.axvspan(c_min,0.5, color='red',alpha=0.08)

    # Adjust the axis boundaries (just cosmetic)
    ax1.axis([c_min, c_max, h_min, h_max])

    # Make labels and title
    plt.title("Classification with scikit-learn")
    plt.xlabel("Classifier output")
    plt.ylabel("Normalized Yields")
    
    # Make legend with smalll font
    legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
    for alabel in legend.get_texts():
        alabel.set_fontsize('small')

    # Save the result to png
    plt.savefig(str(outString)+".pdf")
    plt.savefig(str(outString)+".png")


#def plot_input_variables(X_sig,X_bkg,branch_names,y_bkg=None,n_bins=30,outString=True,plotProcess=True):
def plot_input_variables(X_sig,X_bkg,branch_names,y_bkg=None,n_bins=30,outString=None,plotProcess=None):

    ncolumns = X_sig.size/len(X_sig)
    print ncolumns
    if plotProcess != None:
        X_bkg_2 = X_bkg[np.where(y_bkg==plotProcess),:][0] #this is to plot only one type of process
    else:
        X_bkg_2 = X_bkg

    for i in range(ncolumns):

        sig = X_sig[:,i]
        bkg = X_bkg_2[:,i]

        c_min=min(np.min(d) for d in np.concatenate([X_sig[:,i],X_bkg_2[:,i]]))
        c_max=max(np.max(d) for d in np.concatenate([X_sig[:,i],X_bkg_2[:,i]]))

        #trick to normalize
        weights_sig = np.ones_like(sig)/float(len(sig)) 
        weights_bkg = np.ones_like(bkg)/float(len(bkg)) 

        Histo_S = np.histogram(sig,bins=30,range=(c_min,c_max),weights=weights_sig)
        Histo_B = np.histogram(bkg,bins=30,range=(c_min,c_max),weights=weights_bkg)
        
        bin_edges = Histo_B[1]
        bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
        bin_widths = (bin_edges[1:] - bin_edges[:-1])

        # Lets get the min/max of the Histograms
        AllHistos= [Histo_S,Histo_B]
        h_max = max([histo[0].max() for histo in AllHistos])*1.2
        h_min = min([histo[0].min() for histo in AllHistos])

    

        ax1 = plt.subplot(111)

        ax1.bar(bin_centers-bin_widths/2.,Histo_S[0],facecolor='blue',linewidth=0,width=bin_widths,label='S ',alpha=0.5)
        ax1.bar(bin_centers-bin_widths/2.,Histo_B[0],facecolor='red',linewidth=0,width=bin_widths,label='B ',alpha=0.5)

        # Adjust the axis boundaries (just cosmetic)
        ax1.axis([c_min, c_max, h_min, h_max])
#        ax1.axis([c_min, c_max*0.5, h_min, h_max])
        plt.xlabel(branch_names[i].replace('noexpand:',''))
        plt.ylabel("Normalized Yields")


        plt.savefig(utils.IO.plotFolder+"variableDist"+str(i)+"_"+str(outString)+".eps")
        plt.savefig(utils.IO.plotFolder+"variableDist"+str(i)+"_"+str(outString)+".png")

        plt.show()


def plot_roc_curve(x,y,clf,outString=None):
    decisions = clf.predict_proba(x)[:,clf.n_classes_-1]
    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(y, decisions)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)'%(roc_auc))
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()

    plt.savefig(utils.IO.plotFolder+"rocCurve"+"_"+str(outString)+".eps")
    plt.savefig(utils.IO.plotFolder+"rocCurve"+"_"+str(outString)+".png")
    return fpr,tpr


def plot_roc_curve_multiclass(x,y,clf,classesSchema=[-2,-1,1],classNumber=2,outString=None):#roc curve signal vs all bkg, each one normalized to one
    y=label_binarize(y,classes=classesSchema)
    decisions = clf.predict_proba(x)[:,classNumber]
    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(y[:,classNumber].ravel(), decisions.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)'%(roc_auc))
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()

    plt.savefig(utils.IO.plotFolder+"rocCurve"+"_"+str(outString)+".eps")
    plt.savefig(utils.IO.plotFolder+"rocCurve"+"_"+str(outString)+".png")
    return fpr,tpr

#roc curve signal vs one bkg
def plot_roc_curve_multiclass_singleBkg(x,y,clf,backgroundClassOutput,signalClassOutput=1,outString=None,weights=None):
    x_bkg=np.asarray(x)[np.where(np.asarray(y)==backgroundClassOutput)]
    x_sig=np.asarray(x)[np.where(np.asarray(y)==signalClassOutput)]
    x_tot=np.concatenate((x_bkg,x_sig))

    if weights.all() != None:
        w_bkg = np.asarray(weights)[np.where(np.asarray(y)==backgroundClassOutput)]
        w_sig=np.asarray(weights)[np.where(np.asarray(y)==signalClassOutput)]
        w_tot=np.concatenate((w_bkg,w_sig))
        
    y_bkg=np.asarray(y)[np.where(np.asarray(y)==backgroundClassOutput)]
    y_sig=np.asarray(y)[np.where(np.asarray(y)==signalClassOutput)]
    y_tot=np.concatenate((y_bkg,y_sig))
    
    decisions = clf.predict_proba(x_tot)[:,clf.n_classes_-1]
    # Compute ROC curve and area under the curve
    if weights.all() == None:
        fpr, tpr, thresholds = roc_curve(y_tot.ravel(), decisions,signalClassOutput)
    else:
        fpr, tpr, thresholds = roc_curve(y_tot.ravel(), decisions,signalClassOutput,sample_weight=w_tot.ravel())
        
    roc_auc = auc(fpr, tpr,reorder=True)
        
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)'%(roc_auc))
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    return fpr,tpr



 
def print_roc_report(fpr,tpr,step=0.05):
     print "+===================== ROC report =============+"
     print "| Signal Efficiency | Background Contamination |" 
     print "+----------------------------------------------+"
     for i in range(int(1/step)):
#         print i
         if fpr[np.where((tpr>step*i) & (tpr<step*i+0.005))].size>0:
#             print "True positive rate: "+str(step*i)
#             print "False positive rate:"+str(fpr[np.where((tpr>step*i) & (tpr<step*i+0.005))][0])
             print "|"+str(step*i)+"|"+str(fpr[np.where((tpr>step*i) & (tpr<step*i+0.005))][0])+"|"
     print "============================"



def print_roc_report(fpr,tpr,outString=None,step=0.0001):
#    print "+===================== ROC report ================+"
#    print "|  Signal Efficiency  |  Background Contamination |" 
#    print "+-------------------------------------------------+"
    f= open(utils.IO.plotFolder+str(outString)+"ROC_res.dat","w")
    for i in range(int(1/step)):
#        print i
        index = plotting.bisection(tpr,step*i)
#        print "True positive rate: "+str(step*i)
#        print "False positive rate:"+str(fpr[index])
#        print "|  "+'      %0.4f       '%(step*i)+"|"+'            %0.4f         '%(fpr[index])+"|"
        f.write("%0.4f" %(step*i) + "  " + "%0.4f\n" %(fpr[index]))
    f.close() 

def bisection(array,value):#be careful, works with sorted arrays
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl


        
def plot_feature_importance(clf,features,end='',path='/home/ovtin/cernbox/HHggbb/HHbbggTraining/Training/output_files/'): 
	features_importance =  optimization.feature_importances_(clf)
	features_dict = {features[i]:features_importance[i] for i in range(0,len(features)) }
	sorted_features_values = sorted( list(features_dict.values()),reverse=True)
	sorted_features = [key for key,value in sorted(list(features_dict.items()) ,key=lambda x : x[1], reverse=True)]

	np.save(path+'out_features_values'+end,sorted_features_values)
	np.save(path+'out_features_names'+end,sorted_features)

	y_pos = np.arange(0,len(features)*2,2)  # the x locations for the groups
	width = 1.2       # the width of the bars
	fig,ax = plt.subplots()
	ax.barh(y_pos,sorted_features_values,width,color='green',align='center')
	for x, y in zip(sorted_features_values, y_pos):
 	  ax.text(x + 0.01, y+0.1, np.round(x,3), va='center', fontsize=6,fontweight='bold',color='blue')
	fig.subplots_adjust(left=0.2) 
	ax.set_yticks(y_pos)
	ax.set_yticklabels(sorted_features,rotation=0,fontsize=7)
	ax.invert_yaxis() 
	ax.set_xlabel('F-scores')
	ax.set_xlim(0,sorted_features_values[0]*1.1)
	ax.set_title("Features importance")
	plt.grid()
	plt.savefig(utils.IO.plotFolder+"importanceBar_"+end+".pdf")
	plt.clf()


   
