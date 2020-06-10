
import math
import scipy
import seaborn as sns
from scipy.stats import iqr
import pickle
from tqdm import tqdm


from sklearn.metrics import accuracy_score
from datetime import date
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import interp

from classifiers import *
from data_cleaning import *
from data_transforms import *

def OneVsRest(X_train, y_train,X_test,random_state=42):
    

    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=1500, \
                                                            min_samples_split=2,\
                                                            min_samples_leaf=10,\
                                                            max_depth=5,\
                                                            random_state=random_state))    
                                                            
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    
    
    return y_score




def model_roc(Participants, minlen=10, training=0.7,order=2, sample_size=10,\
              standardise=True, count=True, missing_clean=False, start_average=False,\
              cumsum=True, feedforward=True,naive=False,time=True,class_=None):


    """trying models with different parameters in len(set) or order for roc plot.

    Parameters
    ----------
    Participants: class of participants for the corresponding 2 tests

    minlen_set : list
        size of each participant data.
    order_set: int array or None
        order-size set.
    training : scalar
        Training set proportional.
    sample_size: number for loop
        Default is 50
        
    standardise: data whether or not standardised
        Default True
    count: missing data count or not
        Default True
    missing_clean: rulling out missing data or not
        Default False
    start_average: if the first element is missing, replace it with average or 0
        Default False
    cumsum: if the data is cumumlated 
        Default: True
    naive: using merely mean value of each dimension
    feedforward: True or False
        whether or not the missing values in the piece of data is filled by feedforward function
    time: True or False
        whether or not the time is included 

    class_: on the whole set (None) or on 0/1/2 groups
        default: None
        
    Returns
    -------

    mean_accuracy: average accuracy for each case

    """
    
    from sklearn.preprocessing import label_binarize
    
    
    random.seed(42)
    random_state=42


    n_classes=3

    for i in range(sample_size):
        train_set, test_set = buildData(Participants,\
                                        training=training,\
                                        minlen=minlen,\
                                        class_=class_) 
        
        
        X_train,y_train=data_model(train_set, order,\
                                   minlen=minlen,\
                                   standardise=standardise,\
                                   count=count,\
                                   feedforward=feedforward,\
                                   missing_clean=missing_clean,\
                                   start_average=start_average, \
                                   naive=naive,\
                                   time=time,\
                                   cumsum=cumsum)  
    
        X_test,y_test=data_model(test_set,  order,\
                                 minlen=minlen,\
                                 standardise=standardise,\
                                 count=count,\
                                 feedforward=feedforward,\
                                 missing_clean=missing_clean,\
                                 start_average=start_average, \
                                 naive=naive,\
                                 time=time,\
                                 cumsum=cumsum)   
    
        y_test=label_binarize(y_test, classes=[0, 1, 2])
    
        y_score=OneVsRest(X_train, y_train,X_test)
    
        if i==0:
            y_tests=np.zeros((int(len(y_test)*sample_size),n_classes))
            y_scores=np.zeros((y_tests.shape[0],y_tests.shape[1]))
        
        y_tests[int(i*len(y_test)):int((i+1)*len(y_test)),:]=y_test
        y_scores[int(i*len(y_test)):int((i+1)*len(y_test)),:]=y_score  
        
        
    
    return y_tests, y_scores

def plot_roc(y_test, y_score, title=None,n_classes=3,lw=2):

    """
        plot roc for classification y_score of y_test
    
    Parameters
    ----------
    y_test: true class for test set
    
    y_score: classification score
    
    title: to save as eps
        Default: None (only show but not save it)

    n_classes: int
        The number of classes in total
    
    lw: int
        plot line width
              
    
    """    

    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

   # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC ({0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC ({0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    
    
    for i, color in zip(range(n_classes), colors):
        classes = {0: "BPD", 1: "HC", 2: "BD"}[i]
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC of ' +classes+' ({1:0.2f})'
                 ''.format(i,roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
   # plt.title('Receiver operating characteristic for '+title)
 #   plt.legend(loc="lower right", bbox_to_anchor=(1.8, 0.5))
    plt.legend(loc="lower right")
    if title==None:
        plt.show()  
    else:
        plt.savefig('ROC_for_'+title+'.eps')

