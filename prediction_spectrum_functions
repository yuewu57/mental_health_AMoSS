import os
import random
import numpy as np
import datetime
import time
import csv
import math
import scipy
import seaborn as sns
import h5py
import pickle
from tqdm import tqdm
import copy
import iisignature

from datetime import date

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


from data_cleaning import *
from data_transforms import *
from prediction_functions import *


class Model:
    
    def __init__(self, rf):
        """
        
            The signature-based machine learning model introduced
            in the original paper.

        Parameters
        ----------
            rf : RandomForest Classification
                The trained random forest classifier.

        """

        self.rf=rf

    def test(self,\
             path,\
             order=2,\
             standardise=True,\
             count=True,\
             feedforward=True,\
             missing_clean=False,\
             start_average=False,\
             naive=False,\
             time=False,\
             cumsum=True,\
             feature=int(0)):
        """Tests the model against a particular participant.

        Parameters
        ----------
        path : str
            Path of the pickle file containing the streams
            of data from the participant.
        order : int, optional
            Order of the signature.
            Default is 2.
        
        minlen: int; the length of data considered for each patient.
            Default is 20.
    
        standardise: data whether or not standardised
            Default True
        count: missing data count or not
            Default True
        missing_clean: rulling out missing data or not
            Default False
        start_average: if the firt element is missing, replace it with average or 0
            Default False
        naive: using merely mean value of each dimension

        cumsum: whether or not cumulate the data with time
            Default True
            
        Returns
        -------
        list
            3-dimensional vector indicating how often the participant
            has buckets that were classified in each clinical group.

        """

        # We load the pickle file of the participant
        file = open(path,'rb')
        collection = pickle.load(file)
        file.close()


        # We construct the inputs and outputs to test the model
        
    
        #data handling for classification task   

        x,y =data_model(collection,\
                        order=order,\
                        minlen=minlen,\
                        standardise=standardise,\
                        count=count,\
                        missing_clean=missing_clean,\
                        start_average=start_average, \
                        feedforward=feedforward,\
                        naive=naive,\
                        time=time,\
                        cumsum=cumsum,\
                        feature=int(0))
   

         # We find the predictions corresponding to the computed inputs


        predicted = self.rf.predict(x)
        
        # We find which group the predictions belong to, and
        # store how often the participant belongs to each group
        vector = np.zeros(3)

        for i in range(len(x)):
        
            vector[int(predicted[i])] += 1

            
        
        vector /= float(len(x))
        return vector

    
def train(path,\
          feature=int(1),\
          order=2,\
          minlen=20,\
          standardise=True,\
          count=True,\
          feedforward=True,\
          missing_clean=False,\
          start_average=False, \
          naive=False,\
          time=False,\
          cumsum=True):
    
    """
    
            Trains the model, as specified in the original paper.

        Parameters
        ----------
        path : str
            Path of the pickle file containing the streams
            of data from the participant.
        order : int, optional
            Order of the signature.
            Default is 2.
        
        minlen: int; the length of data considered for each patient.
            Default is 20.
    
        standardise: data whether or not standardised
            Default True
        count: missing data count or not
            Default True
        missing_clean: rulling out missing data or not
            Default False
        start_average: if the firt element is missing, replace it with average or 0
            Default False
        naive: using merely mean value of each dimension

        cumsum: whether or not cumulate the data with time
            Default True

    Returns
    -------
        Model
            Trained model.

    """

    file = open(path,'rb')
    collection = pickle.load(file)
    file.close()
    random_state=42

    # Each clinical group is associated with a point on the
    # plane. These points were found using cross-valiation.

    
    x,y=data_model(collection,\
                   order=order,\
                   minlen=minlen,\
                   standardise=standardise,\
                   count=count,\
                   missing_clean=missing_clean,\
                   start_average=start_average, \
                   feedforward=feedforward,\
                   naive=naive,\
                   time=time,\
                   cumsum=cumsum,\
                   feature=feature)
    
    
    # We train the model using Random Forests.

    reg = OneVsRestClassifier(RandomForestClassifier(n_estimators=1500,\
                                                     random_state=random_state))

    
    reg.fit(x, y)

    # Return the trained model.
    return Model(rf)    

def export(coll,\
           ID,\
           diagnosis_,\
           sample_length=20,\
           test_size=5,\
           path_save="./dataset_spectrum_prediction/"):
    """
    
        Saves as a pickle file the training or testing sets.

    Parameters
    ----------
    coll : list
        List of participants that should be exported. If the
        length of the list is 1, the set is the out-of-sample
        set. Otherwise, it is the training set.
    ID : int
        A random ID that will be used to export the file.
    
    sample_length: Number of observations of each stream of data
    
    test_size: how many piece of sample_length sized data from one patient being tested
        Default: 5
    
    path_save: directory to save the dataset
        
    """

    try:
        os.mkdir(path_save)
        print("Directory " , path_save ,  " Created ") 
    except FileExistsError:
        continue
    
    

    if not os.path.exists(path_save+str(ID)):
            os.makedirs(path_save+str(ID))

            
    dataset=[]
    
        # For each participant and each bucket of appropriate size,
        # add the stream of data to dataset.
    l=copy.deepcopy(coll)
    
    
    if len(coll)==1:
            # We want to export a single participant for
            # testing.
        setType="test_set"
            
            
        for participant in l:
        
            min2=np.minimum(len(participant.data[0]),\
                            len(participant.data[1]))

            random_start=np.random.randint(min2-sample_length-1,\
                                           size=test_size)

            for i in range(test_size):
                p = Participant([participant.data[0][random_start[i]:random_start[i]+sample_length],\
                                 participant.data[1][random_start[i]:random_start[i]+sample_length]],\
                                [participant.time[0][random_start[i]:random_start[i]+sample_length],\
                                 participant.time[1][random_start[i]:random_start[i]+sample_length]],\
                                participant.idNumber,\
                                participant.diagnosis,\
                                [asrm_cutoff(int(participant.data[0][random_start[i]+sample_length])),\
                                 qids_cutoff(int(participant.data[1][random_start[i]+sample_length]))])
            
                dataset.append(p)

            
            
    else:
            # We want to export the rest of the cohort
            # for training.
        setType="train_set"


        for participant in l:
        
            if participant.diagnosis==diagnosis_:
        
                min2=np.minimum(len(participant.data[0]),\
                                len(participant.data[1]))
            
            ########### changing class Participant if change database #############

                random_start=random.randint(0,min2-sample_length-1)

                p = Participant([participant.data[0][random_start:random_start+sample_length],\
                                 participant.data[1][random_start:random_start+sample_length]],\
                                [participant.time[0][random_start:random_start+sample_length],\
                                 participant.time[1][random_start:random_start+sample_length]],\
                                participant.idNumber,\
                                participant.diagnosis,\
                                [asrm_cutoff(int(participant.data[0][random_start+sample_length])),\
                                 qids_cutoff(int(participant.data[1][random_start+sample_length]))])
            
                dataset.append(p)
        
    # Export the dataset.
    filehandler = open(path_save+str(ID)+"/"+setType+".obj","wb")
    pickle.dump(dataset,filehandler)
    filehandler.close()
    

def getCategory(id, path_save="./dataset_spectrum_prediction/"):
    """
        Finds the clinical group a given participant belongs to.

    Parameters
    ----------
    id : int
        ID of the participant.
    
    path_save: directory to save the dataset

    Returns
    -------
    str
        Clinical group that the participant with the given
        ID belongs to.

    """


    file = open(path_save+str(id)+"/test_set.obj",'rb')
    collection = pickle.load(file)
    file.close()

    categories = ["borderline", "healthy", "bipolar"]

    return categories[collection[0].diagnosis]



def get_folders(path_save):
    """Finds all folders in a directory.

    Parameters
    ----------
    a_dir : str
        Directory path.

    Returns
    -------
    list of str
        List of all folders in the directory.

    """

    return [name for name in os.listdir(path_save)
            if os.path.isdir(os.path.join(path_save, name))]


def trim_triangle(col,index=1):
    
    """trim healthy data such that plot can be seen.

    Parameters
    ----------
    col : a collection of healthy data

    Returns
    -------
    list of str
        List of data can has been trim by threshold 0.03.

    """
   

    try1=copy.deepcopy(col)

    for md in try1:
        if md[0]==0.0:
            if md[int(index)]<0.95:
                md[0]=0.03

                md[2]-=0.03
 
    return try1


def plotDensityMap(scores,label_='depressed',title_="BD"):
    """Plots, given a set of scores, the density map on a triangle.

    Parameters
    ----------
    scores : list
        List of scores, where each score is a 3-dimensional list.

    """


    TRIANGLE = np.array([[math.cos(math.pi*0.5), math.sin(math.pi*0.5)],
                        [math.cos(math.pi*1.166), math.sin(math.pi*1.166)],
                        [math.cos(math.pi*1.833), math.sin(math.pi*1.833)]])

        
    pointsX = [score.dot(TRIANGLE)[0] for score in scores]
    pointsY = [score.dot(TRIANGLE)[1] for score in scores]

    vertices = []
    vertices.append(np.array([1,0,0]).dot(TRIANGLE))
    vertices.append(np.array([0,1,0]).dot(TRIANGLE))
    vertices.append(np.array([0,0,1]).dot(TRIANGLE))
    for i in range(3):
        p1 = vertices[i]
        if i == 2:
            p2 = vertices[0]
        else:
            p2 = vertices[i+1]
        c = 0.5 * (p1 + p2)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linestyle='-', linewidth=2)
        plt.plot([0, c[0]], [0, c[1]], color='k', linestyle='-', linewidth=1)



    ax = plt.gca()
    ax.set_xlim([-1.2, 1.32])
    ax.set_ylim([-0.7,1.3])

    ax.text(0.8, -0.6, label_)
    ax.text(-1.1, -0.6, 'normal')
    ax.text(-0.15, 1.05, 'No Answer')


    data = [[pointsX[i], pointsY[i]] for i in range(len(pointsX))]

    H, _, _=np.histogram2d(pointsX,pointsY,bins=40,normed=True)
    norm=H.max()-H.min()
  
    contour1=0.75
    target1=norm*contour1+H.min()
    def objective(limit, target):
        w = np.where(H>limit)
        count = H[w]
        return count.sum() - target


    level1 = scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
    levels = [level1]

    data = np.array(data)

    sns.kdeplot(np.array(pointsX), np.array(pointsY), shade=True, ax=ax)
    sns.kdeplot(np.array(pointsX), np.array(pointsY), n_levels=3, ax=ax, cmap="Reds")
    plt.show()
