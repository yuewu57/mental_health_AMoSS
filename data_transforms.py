from data_cleaning import *


def standardise1(x,l1,l2):
    """standardise data x to [-1,1]


    Parameters
    
    x: original data    
    l1: lower bound
    l2: upper bound
    ----------

    Returns
    -------
    
    standardised data


    """    
    x-=l1
    x*=2/float(l2-l1)
    x-=1 
    return x

def standardise(x,l1,l2):
    """standardise data x to [0,1]


    Parameters
    
    x: original data    
    l1: lower bound
    l2: upper bound
    ----------

    Returns
    -------
    
    standardised data


    """  
   
    return (x-l1)/float(l2-l1)


def normalise(data,cumsum=True,count=True,time=True):
    """Normalises the data of the patient with missing count.

    Parameters
    ----------
    data : two dim data, consisting of ALTMAN and QIDS scores


    Returns
    -------
    normalised_data: data that are normalised and cumulated.

    """

    normalised_data=np.zeros((data.shape[0],data.shape[1]))

    scoreMAX=[20,27]
    scoreMIN=[0,0]

    if count:
        if time:
            len_data=data.shape[1]-2
        else:
            len_data=data.shape[1]-1
    else:
        len_data=data.shape[1]

    for i in range(len_data):

        for j in range(data.shape[0]):
            
            normalised_data[j][i]=standardise1(data[j][i],scoreMIN[i],scoreMAX[i])

            if j>0:
                    normalised_data[j][i]+=normalised_data[j-1][i]


    if count:
        if time:
            for kk in range(len(data[:,-1])):
                normalised_data[kk][-1]=standardise(data[kk][-1],0,data[-1][-1])
            
            if cumsum and data[-1][-2]!=0:
                for jj in range(len(data[:,-2])):
                    normalised_data[jj][-2]=standardise(data[jj][-2],0,data[-1][-2])

        else:
            if cumsum and data[-1][-1]!=0:
                for jjj in range(len(data[:,-1])):
                    normalised_data[jjj][-1]=standardise(data[jjj][-1],0,data[-1][-1])
                    
#         if data[0][-1]!=data[-1][-1]:
#             for j in range(len(data[:,-1])):
#                 normalised_data[j][-1]=standardise(data[j][-1],data[0][-1],data[-1][-1])
#         if time:
#      #       if data[-1][-2]!=0:
#                 for jj in range(len(data[:,-2])):
#                     normalised_data[jj][-2]=standardise(data[jj][-2],data[0][-2],data[-1][-2])
   
    
    return normalised_data


def clean_missing1(data):

    """clean missing data with strategy 1.


    Parameters
    ----------
     data:  orignal data in list form.



    Returns
    -------
    newdata: data without missing one in array form.

    """


    num_col=len(data)
    newdata_list=[]
    
    for i in range(num_col):
        newdata_list.append(np.delete(data[i],np.where(data[i]==-1)))

    num_len=[len(newdata_list[j]) for j in  range(num_col)]

    if 0 in num_len:
        return False
    else:

        num_min=np.min(num_len)

        newdata=np.zeros((num_min,num_col))
        for jj in range(num_col):

            newdata[:,jj]=newdata_list[jj][:num_min]
                

        return newdata

def list_to_num(data):

    type_num=len(data)

    lendata=10000
    for i in range(type_num):
        lendata=np.minimum(len(data[i]),lendata)



    newdata=np.zeros((lendata,type_num)).astype(int)

    for j in range(type_num):
        newdata[:,j]=[data[j][t] for t in range(lendata)]

    return newdata


def  feed_forward(data,count=True, cumsum=True, start_average=False,time=True):
    """counting missing data with feedforward strategy as in Andrey paper.


    Parameters
    ----------
     data:  orignal data in list form. (2 streams)



    Returns
    -------
    newdata: data with missing count (3 dim) in array form.

    """

    lendata=len(data[0])

    if count:
        if time:
            newdata=np.zeros((lendata,len(data)+2)).astype(int)
        else:
            newdata=np.zeros((lendata,len(data)+1)).astype(int)
    else:
         newdata=np.zeros((lendata,len(data))).astype(int)   
    
    for j in range(len(data)):
        
        newdata[:,j]=[data[j][t]for t in range(lendata)]


    if start_average:
        for jj in range(len(data)):                        
            if newdata[0,jj]==-1:
                newdata[0,jj]=5 #just below signaficant symptoms of mania
                if count:
                    if time:
                        newdata[0,-2]+=1
                    else:
                        newdata[0,-1]+=1

    else:
        for jj in range(len(data)):
            if newdata[0,jj]==-1:
                newdata[0,jj]=0
                if count: 
                    if time:
                        newdata[0,-2]+=1
                    else:
                        newdata[0,-1]+=1     
                        

    for jjj in range(len(data)):
        for i in np.where(newdata[:,jjj]==-1)[0]:
            newdata[i,jjj]=newdata[i-1,jjj]
            if count:
                if time:
                    newdata[i,-2]+=1
                else:
                    newdata[i,-1]+=1



    if count:
        
        for t in range(lendata)[1:]:
            if time:
                if cumsum:
                    newdata[t,-2]+=newdata[t-1,-2]
                newdata[t,-1]=t
            else:
                if cumsum:
                    newdata[t,-1]+=newdata[t-1,-1]

    return newdata


