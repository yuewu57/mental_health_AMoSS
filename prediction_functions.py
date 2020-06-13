import os
import random
import h5py
import time
import csv
import math
import scipy
import copy
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import iisignature

from datetime import date


from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from data_cleaning import *
from data_transforms import *



def data_model(collection,\
               minlen=20,\
               order=2,\
               standardise=True,\
               count=True,\
               feedforward=True,\
               missing_clean=False,\
               start_average=False, \
               naive=False,\
               time=True,\
               cumsum=True,\
               feature=int(0)):

    
    """
    
          process data before fitting into machine learning models.
          
          
    Parameters
    ----------
    collection : list
        The out-of-sample set.
    order : int, optional
        Order of the signature.
        Default is 2.
        
    minlen: int
        the length of data considered for each patient.
        Default is 20.
    
    standardise: True or False
        whether or not the piece of data being standardised
    count: True or False
        whether or not the piece of data being counted for missing value
    
    feedforward: True or False
        whether or not the missing values in the piece of data is filled by feedforward function
        
    missing_clean:True or False
        whether or not the missing values in the piece of data is to be cleaned
    
    start_average: True or False
        whether or not the missing values at the beginning of the piece of data is filled by the average
    naive: True or False
        whether or not the navid method (by calculating the mean) is implemented
    time: True or False
        whether or not the time is included 
    
    cumsum: True or False
        whether or not the piece of data is being cumulated
        
    feature: which group we are considering
        Default: int(0), otherwise, int(1)/int(2)
        
    Returns
    -------
    
    x,y in appropriate form
    
    """




    x=[]
    y=[]


    for participant in collection:

        if dual:
            par_data=participant.data
        else:
            par_data=[participant.data[feature]]


        if missing_clean:
                participant_data=clean_missing1(par_data)
        elif count:
            if standardise:

                participant_data=normalise(feed_forward(par_data,\
                                                        count=count,\
                                                        time=time,\
                                                        start_average=start_average,\
                                                        cumsum=cumsum),\
                                           count=count,\
                                           time=time)

            else:

                participant_data=feed_forward(par_data,\
                                              count=count,\
                                              time=time,\
                                              start_average=start_average,\
                                              cumsum=cumsum)

        else:
            if standardise:
                if feedforward:
                    participant_data=normalise(feed_forward(par_data,\
                                                            count=count,\
                                                            time=time,\
                                                            start_average=start_average,\
                                                            cumsum=cumsum),\
                                               count=count,\
                                               time=time)

                else:
                    participant_data=normalise(list_to_num(par_data),\
                                               count=count,\
                                               time=time)
            else:
                if feedforward:
                        participant_data=feed_forward(par_data,\
                                                      count=count,\
                                                      time=time,\
                                                      start_average=start_average,\
                                                      cumsum=cumsum)
                else:
                    participant_data=list_to_num(par_data)



        

        if participant_data is not False:

            if naive:
                if len(participant_data)==0:
                        x.append([-1])

                else:

                    x.append([np.sum(participant_data)/minlen])
                    
            else:
                     x.append(iisignature.sig(participant_data, order))
            
            y.append(participant.nextdata[feature])




    return x,y




def rf_nextdaymodel(X_train, y_train,X_test,\
                    regression=False,\
                    feature=int(0),\
                    random_state=42):
    
    """
    
          random forest base model for next report prediction.
          
          
    Parameters
    ----------
    
    X_train, y_train, X_test: the lists of train input, train output and test input
    
    regression: whether this prediction is based on regressor
      Default: False (use classifier), otherwise True
    
    feature: consider which group from 0/1/2
      Default: int(0)
      
    random state: Default 42
    
    
    Return
    ----------
    the list of test prediction
    
    """
    
    if regression:
        

        regressor=RandomForestRegressor(n_estimators=1500,\
                                        random_state=random_state)
        y_predicted = regressor.fit(X_train,\
                                    y_train).predict(X_test)
        
    else:
    
        classifier =OneVsRestClassifier(RandomForestClassifier(n_estimators=1500, \
                                                               random_state=random_state))
        y_predicted = classifier.fit(X_train, \
                                     y_train).predict(X_test)

    return y_predicted




def accuracy_(x,y):

    mean=0
    for i in range(len(x)):

        if int(x[i])==int(y[i]):
              mean+=1


    return mean/len(x)

def MSE(c,d,feature=int(0)):
    """    
          Computing the mean squared error for two lists of lists c and d
 
    """    
    a = [item for sublist in c for item in sublist]
    b= [item for sublist in d for item in sublist]     
    
    if len(a)!=len(b):
        print("something is wrong.")
    else:

        sd=np.array([(a[i]-b[i])**2 for i in range(len(a))])
        return np.sqrt(np.mean(sd))
 
 
def MAE(c,d,feature=int(0),scaling=False):
    """    
          Computing the mean absolute error for two lists of lists c and d
 
    """        
    a = [item for sublist in c for item in sublist]
    b= [item for sublist in d for item in sublist]  
    
    if not scaling:
        a=scaling_list(a1,feature=feature)
        b=scaling_list(b1,feature=feature)
  
    if len(a)!=len(b):
        print("something is wrong.")
    else:
        
        sd=np.array([np.abs(a[i]-b[i]) for i in range(len(a))])
        
        return len(np.where(sd<1)[0])/len(sd), np.mean(sd)
    

def R2(c,d,feature=int(0)):
    """    
          Computing the mean r2 error of two lists of lists c and d
 
    """          
    a=np.array([r2_score(c[i], d[i]) for i in range(len(c))])

    return np.mean(a)
    



    
def comprehensive_model(Participants,\
                        class_,\
                        minlen=10,\
                        training=0.7,\
                        sample_size=10,\
                        cumsum=True):


    """
            trying models (stateMRSPM, level 2, naive model) with different parameters in len(set) or order in one-go.

    Parameters
    ----------
    Participants: class of participants for the corresponding 2 tests
    
    class_: which class we are working on (0/1/2)
    
    minlen_set : list
        size of each participant data.

    training : scalar
        Training set proportional.
        
    sample_size: number for loop
        Default is 50
    
    cumsum: a parameter in data-transforming step, can be set to True for all models

    Returns
    -------

    mean_accuracy: average accuracy for each case

    """


    random.seed(42)
    
    random_state=42

    standardise_set=[False, True]
    count_set=[False, True]
    feedforward_set=[False, True]
    naive_set=[True,  False]
    time_set=[False,  True]
    missing_clean_set=[False,False]

    order_set=[None,int(3)]

    feature_set=[int(0),int(1)]
    
    y_collections=[[] for i in range(int(len(feature_set)*len(standardise_set)))]
    y_pred_collections=[[] for i in range(int(len(feature_set)*len(standardise_set)))]
    
    accuracy=np.zeros((len(feature_set),len(standardise_set)))
    mse=np.zeros((len(feature_set),len(standardise_set)))
    mae=np.zeros((len(feature_set),len(standardise_set)))
    
    for i in range(sample_size):
      
        train_set, test_set = buildData_prediction(Participants,\
                                                   training=training,\
                                                   minlen=minlen,\
                                                   class_=class_,\
                                                   regression=regression)

        for j in range(len(standardise_set)):

            for ii in range(len(feature_set)):

                X_train,y_train=data_model(train_set, \
                                           order=order_set[j],\
                                           standardise=standardise_set[j],\
                                           count=count_set[j],\
                                           feedforward=feedforward_set[j],\
                                           missing_clean=missing_clean_set[j],\
                                           time=time_set[j],\
                                           naive=naive_set[j],\
                                           cumsum=cumsum,\
                                           feature=feature_set[ii])


                X_test,y_test=data_model(test_set,\
                                         order=order_set[j],\
                                         standardise=standardise_set[j],\
                                         count=count_set[j],\
                                         feedforward=feedforward_set[j],\
                                         missing_clean=missing_clean_set[j],\
                                         time=time_set[j],\
                                         naive=naive_set[j],\
                                         cumsum=cumsum,\
                                         feature=feature_set[ii])

                current_index=int(j*len(feature_set)+ii)
                
    
                y_test_pred_=rf_nextdaymodel(X_train,y_train,X_test,\
                                             regression=regression,\
                                             feature=feature_set[ii] )

                    
                y_pred_collections[current_index].append(y_test_pred_)
                y_collections[current_index].append(y_test)
                accuracy[ii,j]+=accuracy_(y_test, y_test_pred_)/sample_size




    return accuracy
  
  
def comprehensive_nomissing_model(Participants,\
                                  class_,\
                                  minlen=10,\
                                  training=0.7,\
                                  sample_size=10,\
                                  scaling=True,\
                                  cumsum=True):


    """trying models with different parameters in len(set) or order.

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
    start_average: if the firt element is missing, replace it with average or 0
        Default False
        concatenation: cancatenated with absolute value of initial values
        Default False
    naive: using merely mean value of each dimension
    hour: adding hour/24

    Returns
    -------

    mean_accuracy: average accuracy for each case

    """


    random.seed(42)
    
    random_state=42

    standardise_set=[False, True, True]
    count_set=[False, True, True]
    feedforward_set=[False, True, True]
    naive_set=[True,  False,  False]
    time_set=[False,  True,  True]
    missing_clean_set=[False,False,False]

    order_set=[None,int(2),int(3)]

    feature_set=[int(0),int(1)]
    
    y_collections=[[] for i in range(int(len(feature_set)*len(standardise_set)))]
    y_pred_collections=[[] for i in range(int(len(feature_set)*len(standardise_set)))]
    
    accuracy=np.zeros((len(feature_set),len(standardise_set)))
    mse=np.zeros((len(feature_set),len(standardise_set)))
    mae=np.zeros((len(feature_set),len(standardise_set)))
    r2=np.zeros((len(feature_set),len(standardise_set)))
    
    for i in range(sample_size):
        train_set, test_set = buildData_prediction_nomissing(Participants,\
                                                             training=training,\
                                                              minlen=minlen,\
                                                             class_=class_)

        for j in range(len(standardise_set)):

            for ii in range(len(feature_set)):

                X_train,y_train=data_model(train_set,\
                                           order=order_set[j],\
                                           standardise=standardise_set[j],\
                                           count=count_set[j],\
                                           feedforward=feedforward_set[j],\
                                           missing_clean=missing_clean_set[j],\
                                           time=time_set[j],\
                                           naive=naive_set[j],\
                                           feature=feature_set[ii],\
                                           cumsum=cumsum)


                X_test,y_test=data_model(test_set,\
                                         order=order_set[j],\
                                         standardise=standardise_set[j],\
                                         count=count_set[j],\
                                         feedforward=feedforward_set[j],\
                                         missing_clean=missing_clean_set[j],\
                                         time=time_set[j],\
                                         naive=naive_set[j],\
                                         feature=feature_set[ii],\
                                         cumsum=cumsum)
                
                current_index=int(j*len(feature_set)+ii)
                

   
                y_test_pred_=rf_nextdaymodel(X_train,y_train,X_test,\
                                             regression=True,\
                                             feature=feature_set[ii])
                
                if scaling:
                    
                    y_test_pred_=scaling_list(y_test_pred_,\
                                              feature=feature_set[ii])
                    y_test=scaling_list(y_test,\
                                        feature=feature_set[ii])                
                    
                y_pred_collections[current_index].append(y_test_pred_)
                
                y_collections[current_index].append(y_test)


                
    for j in range(len(standardise_set)):

        for ii in range(len(feature_set)):
            current_index=int(j*len(feature_set)+ii)

            mse[ii,j]=MSE(y_pred_collections[current_index],\
                          y_collections[current_index],\
                          feature=feature_set[ii])
            
            accuracy[ii,j],mae[ii,j]=MAE(y_pred_collections[current_index],\
                                         y_collections[current_index],\
                                         feature=feature_set[ii])
            
            r2[ii,j]=R2(y_pred_collections[current_index],\
                        y_collections[current_index],\
                        feature=feature_set[ii])

    return accuracy, mse, mae, r2


