from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
import iisignature

from sklearn.svm import SVC
from sklearn.svm import SVR


from data_cleaning import *
from data_transforms import *



def distance(p,p0):
    return np.linalg.norm(p0-np.array(p))




def accuracy1(x,y):
    mean=0
    for i in range(len(x)):
        if x[i][0]==y[i][0] and x[i][1]==y[i][1]:
            mean+=1
      
    return mean/len(x)


def accuracy(x,y):
    mean=0
    for i in range(len(x)):
        if x[i]==y[i]:
            mean+=1
    return mean/len(x)


def data_model(collection, order=2,minlen=20, standardise=True, count=True, feedforward=True,\
               missing_clean=False,start_average=False, naive=False,time=True,cumsum=True):


    
    """process data before fitting into machine learning models.

    Parameters
    ----------
    collection : list
        The out-of-sample set.

    threshold : array
    order : int, optional
        Order of the signature.
        Default is 2.

    Returns
    -------
    
    x,y in appropriate form
    
    """


    x=[]
    y=[]


    for participant in collection:


        par_data=participant.data


        if missing_clean:
                participant_data=clean_missing1(par_data)
        elif count:
            if standardise:

                participant_data=normalise(feed_forward(par_data,\
                count=count, time=time,start_average=start_average,cumsum=cumsum),\
                cumsum=cumsum, count=count,time=time)

            else:

                participant_data=feed_forward(par_data,\
                count=count, time=time,start_average=start_average,cumsum=cumsum)

        else:
            if standardise:
                if feedforward:                
                    participant_data=normalise(feed_forward(par_data,\
                        count=count,time=time,start_average=start_average,cumsum=cumsum),\
                        cumsum=cumsum, count=count,time=time)

                else:
                    participant_data=normalise(list_to_num(par_data),count=count,time=time,cumsum=cumsum)
            else:
                if feedforward:
                        participant_data=feed_forward(par_data,count=count,time=time,\
                                                      start_average=start_average,cumsum=cumsum)
                else:
                    participant_data=list_to_num(par_data)
        
        
         
        if participant_data is not False:
            


            if naive:
                if missing_clean:

                    x.append(np.sum(participant_data,axis=0)/minlen)
                else:

                        
                    x.append(np.sum(list_to_num(par_data),axis=0)/minlen)

            else:


                    x.append(iisignature.sig(participant_data, order))
            



            y.append(participant.diagnosis)


    return x,y


def rf_model(X_train,y_train,X_test,y_test,random_state = 42):
    """ random forest model

       Parameters
    ----------
    X_train : training set

    y_train : training class

    X_test : test set

    y_test: test class

    threshold: triangle coordinates

    Returns
    -------

    x,y in appropriate form


    """
        

                 
    gridF=OneVsRestClassifier(RandomForestClassifier(n_estimators=1500,min_samples_leaf=10,max_depth=5,\
                                      min_samples_split=2, random_state = random_state))



    gridF.fit(X_train, y_train)
    predicted = gridF.predict(X_test)
    CM = confusion_matrix(y_test, predicted)
    return CM, accuracy(predicted, y_test)
    
    
    
    
def rf_cv(X_train,y_train,X_test,y_test,random_state = 42):
    """ random forest model

       Parameters
    ----------
    X_train : training set

    y_train : training class

    X_test : test set

    y_test: test class

    threshold: triangle coordinates

    Returns
    -------

    x,y in appropriate form




    """

    forest = RandomForestClassifier(random_state = random_state)        

    n_estimators = [100, 400, 800, 1000,1500,2000]
    max_depth = [5, 8, 15, 25, 30,None]
    min_samples_split = [2, 5, 10, 15]
    min_samples_leaf = [1, 2, 5, 10]
    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,
             min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf)

    gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, n_jobs = -1)    
    gridF.fit(X_train, y_train)
    
    best_grid = gridF.best_estimator_
    print(best_grid)            
    predicted = best_grid.predict(X_test)
    
    return accuracy(predicted, y_test)
    
def model_onego(Participants, minlen, training=0.7, sample_size=50, \
                start_average=False, cv=False,cumsum=True, class_=None):

    """trying models with different parameters in len(set) or order.

     Parameters
     ----------
     Participants: class of participants for the corresponding 2 tests

     minlen: num
         size of each participant data.
     training : scalar
         Training set proportional.
     sample_size: number for loop
         Default is 50
     start_average: if the firt element is missing, replace it with average or 0
         Default False



     Returns
     -------

     mean_accuracy: average accuracy for each case

    """
     
    

    random.seed(42)

    
    standardise_set=[False,True, True]
    count_set=[False, True,True]
    feedforward_set=[False,True,True]
    naive_set=[True, False,False]
    time_set=[False, True,True]
    missingclean_set=[False, False, False]


    order_set=[None, 2,3]
     

    col_num=len(standardise_set)

        
    mean_accuracy=np.zeros(col_num)
    

    
    mean_CM=[np.zeros((3,3)) for j in  range(col_num)]
    
    accuracy_collections=[[] for j in  range(col_num)]
  
 
    for i in range(sample_size):


            #####################################################################################
            #################### Test on pair of sum scores from ALTMAN+QIDS ####################
             #####################################################################################

                train_set, test_set = buildData(Participants, training=training,\
                                                minlen=minlen,class_=class_)



                for j in range(col_num):



                
                    X_train,y_train=data_model(train_set,minlen=minlen,\
                                               order=order_set[j],\
                                               standardise=standardise_set[j],\
                                               count=count_set[j],\
                                               missing_clean=missingclean_set[j], \
                                               start_average=start_average,\
                                               feedforward=feedforward_set[j],\
                                               naive=naive_set[j],\
                                               time=time_set[j],\
                                               cumsum=cumsum)
                    


                    X_test,y_test=data_model(test_set,minlen=minlen, \
                                             order=order_set[j],\
                                             standardise=standardise_set[j], \
                                             count=count_set[j],\
                                             missing_clean=missingclean_set[j], \
                                             start_average=start_average,\
                                             feedforward=feedforward_set[j],\
                                             naive=naive_set[j],\
                                             time=time_set[j],\
                                             cumsum=cumsum)

                    if cv:
                        accuracy= rf_cv(X_train,y_train,X_test,y_test,threshold)
                    else:

                        CM,accuracy= rf_model(X_train,y_train,X_test,y_test,threshold,regression=regression)
                            
                        mean_CM[j]+=CM
                        
                    mean_accuracy[j]+=accuracy/sample_size
                    accuracy_collections[j].append(accuracy)
                    


        
        return mean_CM, mean_accuracy,accuracy_collections

    



