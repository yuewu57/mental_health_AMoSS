import os
import csv
import random
import numpy as np
import h5py
import copy
import matplotlib.dates as mdates
from datetime import date

class Participant:
    
        def __init__(self, data, time, id_n,diagnosis,nextdata):

            self.idNumber = id_n
            self.data = data
            self.time=time
            self.diagnosis=diagnosis
            self.nextdata=nextdata

            
def loadParticipants(path):
    """Loads the participant cohort.
    
    Parameters
    ----------
    path : str
        Location of the directory with the data (original data stored in mat file).

    Returns
    -------
    list of Participant for ALTMAN test, time list, id list, QS list
    
    list of Participant for QIDS test, time list, id list, QS list
    
     list of Participant for both ALTMAN & QIDS test, time list, id list, QS list

    """

    
    participants_list=[]
    participants_dataALTMAN=[]


    participants_timeALTMAN=[]

    
    participants_dataQIDS=[]

    participants_timeQIDS=[]
    
    for filename in sorted(os.listdir(path)):
        f = h5py.File(path+filename, 'r')
        
        k = list(f.keys())
        
        if "QIDS" in f['data'] and "ALTMAN" in f['data']:
            participants_list.append(filename.split("-")[0])
            participants_dataALTMAN.append(f[k[1]]['ALTMAN']['data'][()][0])
            participants_dataQIDS.append(f[k[1]]['QIDS']['data'][()][0])

            participants_timeALTMAN.append(f[k[1]]['ALTMAN']['time'][()][0])
            participants_timeQIDS.append(f[k[1]]['QIDS']['time'][()][0])   
          
            

            
    participant_data_list=[]
    participant_data_list.append(participants_dataALTMAN)
    participant_data_list.append(participants_dataQIDS)

    
    participants_time_list=[]
    participants_time_list.append(participants_timeALTMAN)
    participants_time_list.append(participants_timeQIDS)

    
    return participants_list,participant_data_list,participants_time_list

def make_classes(participants_data_list,participants_time_list,participants_list):
    """data process to make class Participant

    Parameters
    
    List of corresponding test1 & test2, test1 time & test 2 time, id list
    ----------

    Returns
    -------
    class of participants for the corresponding 2 tests


    """
    
    num=len(participants_list)
    
    Participants=sorted(list(csv.reader(open("./source_data/patients.csv"))))
    
    participants=[]
    
    t=0
    
    for i in range(num):

        n=int(participants_list[i])
        
        for l in Participants:
            if int(l[0])==n:
                
                if not l[1].isdigit():

                    print(n)
                    
                    break
                    

                data=[participants_data_list[j][i].astype(int) for j in range(len(participants_data_list)) ]

                time=[participants_time_list[j][i].astype(int) for j in range(len(participants_time_list)) ]
                
                                
                                
                bp0 = int(l[1])
                
                bp = {1: 2, 2: 0, 3: 1}[bp0]
                
                participant=Participant(data, time, n, bp,None)
                participants.append(participant)
                t+=1
                break

            
    return participants

def cleaning_same_data(Participants):
    
    """cleaning redundant data: if two data are stored in the same day,
    and scores are the same, keep one; if one score is recorded as missing, then keep the
    other one.


    Parameters
    
    class participant data
    ----------

    Returns
    -------
    
    shortened participant data


    """    
    

    
    Pars=copy.deepcopy(Participants)
    
    n=len(Pars[0].data)
    
    for par in Pars:
        for i in range(n):
            
            t=0
            
            total_len=len(par.time[i])

            
            for j in range(total_len)[:-1]:

                while int(par.time[i][j+1])==int(par.time[i][j]):
                    
                    if par.data[i][j+1]==par.data[i][j]:
                        par.data[i]=np.delete(par.data[i],j+1)
                            
                        par.time[i]=np.delete(par.time[i],j+1)
                        t+=1
                        
                    elif par.data[i][j]<0:


                        
                        par.data[i]=np.delete(par.data[i],j)    
                        par.time[i]=np.delete(par.time[i],j)
                        
                        t+=1
                        
                        
                    else:
                        if par.data[i][j+1]<0:
                            par.data[i]=np.delete(par.data[i],j+1) 
                            par.time[i]=np.delete(par.time[i],j+1) 
                            
                            t+=1
                        else:
                            break
                        
                    if int(j+1+t)==total_len:
                        break
                        

                if int(j+2+t)>=total_len:
                    break
                
    
    return Pars    

def dates_difference(s1,s2):
    
    d1=date(mdates.num2date(s1).year,mdates.num2date(s1).month,mdates.num2date(s1).day)
                                          
    d2=date(mdates.num2date(s2).year,mdates.num2date(s2).month,mdates.num2date(s2).day)
    
    delta = d2 - d1
                                          
    return delta.days
   
    
    
def cleaning_sameweek_data(Participants):
    
    Pars=copy.deepcopy(Participants)
    
    n=len(Pars[0].data)
    
    for par in Pars:
        
        start_sets=[int(mdates.num2date(par.time[ji][0]).isocalendar()[1]) for ji in range(n)]
        
        start_week=int(np.min(start_sets))
        if len(np.where(start_sets==start_week)[0])!=0:
            start_n=np.where(start_sets==start_week)[0][0]
        else:
            start_n=int(0)
            
        for ji in range(n):
            len_start=int(start_sets[ji]-start_week)
            

            
            if len_start!=0:
                
                par.time[ji]=np.append(par.time[start_n][:len_start],par.time[ji])
                
                par.data[ji]=np.append(np.zeros(len_start,dtype=int)-int(1),par.data[ji])
            
        for i in range(n):
            
            total_len=len(par.time[i])


            t=0
            for j in range(total_len)[:-1]:
                
                while dates_difference(par.time[i][j], par.time[i][j+1])<3:

###### Older version; can be reactive if the current version is not so nice########                   
#                     if par.data[i][j]<0 and par.data[i][j+1]>-1:
                        
#                         par.data[i]=np.delete(par.data[i],j)
#                         par.time[i]=np.delete(par.time[i],j)
#                     else:
#                         par.data[i]=np.delete(par.data[i],j+1)                           
#                         par.time[i]=np.delete(par.time[i],j+1)
                    
                    if par.data[i][j]<0: 

                        par.data[i]=np.delete(par.data[i],j)
                        par.time[i]=np.delete(par.time[i],j)
                    else:
                        if par.data[i][j+1]<0:
                            par.data[i]=np.delete(par.data[i],j+1)                           
                            par.time[i]=np.delete(par.time[i],j+1)                    
                        else:
                            par.data[i][j]=(par.data[i][j]+par.data[i][j+1])/2
                            par.data[i]=np.delete(par.data[i],j+1)                           
                            par.time[i]=np.delete(par.time[i],j+1) 
                    t+=1

                    if int(j+1+t)==total_len:
                        break
                    
                        

                
                if int(j+2+t)>=total_len:
                     break
                
    
    return Pars

def buildData(collection, training=0.7,minlen=20,class_=None):
    
    """Builds the training and out-of-sample sets.
    
    
    Parameters
    ----------
     collection :  data is located.
     training : float, optional
        Percentage of the data that will be used to
        train the model.
        Default is 0.7.

    minlen: num
        The least data length, default 20.
    
    class_: build data on class 0/1/2 only, otherwise, Nonw
    
    Returns
    -------
    list
        Training set.
    list
        Out-of-sample set.
    
    """

    
    collection1=copy.deepcopy(collection)
    len_data=len(collection1[0].data)
    
    
    total_len=len(collection1)

    jj=0
    
    t=0
    
    
    while jj<total_len:
        
        min2=np.min([len(collection1[jj].data[i]) for i in range(len_data)])
        
        while min2<=minlen:

            collection1.remove(collection1[jj])
            total_len-=1
            t+=1

            min2=np.min([len(collection1[jj].data[i]) for i in range(len_data)])
            

                  
            
        random_start=random.randint(0,min2-minlen)

        for i in range(len_data):
                collection1[jj].data[i]=collection1[jj].data[i][random_start:random_start+minlen]
                
                collection1[jj].time[i]=collection1[jj].time[i][random_start:random_start+minlen]
                
        jj+=1

                
    collection2=[]
    
    if class_ is not None:
        for par in collection1:
            if par.diagnosis==class_:
                collection2.append(par)
    else:
        collection2=collection1
        
  
    random.shuffle(collection2)
    training_set = collection2[:int(training*len(collection2))]

    out_of_sample = collection2[int(training*len(collection2)):]
    
    
    return training_set, out_of_sample

