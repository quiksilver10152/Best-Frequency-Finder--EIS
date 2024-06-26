# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:22:46 2024

@author: GPZ1100

Module Function
    This module accepts an .xlsx file exported from Gamry and calculates the most important frequency points using 2 methods:
        
        Method 1: Creates every possible unique combination of 5 data points and runs against an Adaboost prediction algorithm.
        Returns the best ensemble of frequencies for each ECM circuit model.
        
        Method 2: Deletes a single datapoint and runs against an Adaboost model. Returns the frequency that caused the biggest 
        drop in prediction score.

"""

# --------------------------------------------------------- Enter Experiment Info Below  ---------------------------------------------------------

#Yes, the r in front of the quotations is NECESSARY!
filePath = r"C:\Users\quiks\Downloads\EISdata.xlsx"
trialNum = 0
#If using different frequencies, this needs to be updated.
eisFreqs = [1.00E+05,7.95E+04,6.31E+04,5.02E+04,3.99E+04,3.16E+04,2.52E+04,2.00E+04,1.59E+04,1.26E+04,1.01E+04,
            8.02E+03,6.33E+03,5.02E+03,3.98E+03,3.17E+03,2.53E+03,1.98E+03,1.58E+03,1.27E+03,9.98E+02,
            7.97E+02,6.28E+02,5.06E+02,3.98E+02,3.16E+02,2.52E+02,1.99E+02,1.58E+02,1.26E+02,1.00E+02,
            7.90E+01,6.33E+01,4.99E+01,3.97E+01,3.17E+01,2.49E+01,1.99E+01,1.58E+01,1.24E+01,9.93E+00,
            7.95E+00,6.32E+00,5.01E+00,3.95E+00,3.16E+00,2.50E+00,2.00E+00,1.59E+00,1.27E+00,9.99E-01,
            7.92E-01,6.33E-01,5.04E-01,4.01E-01,3.17E-01,2.52E-01,2.00E-01,1.59E-01,1.26E-01,1.00E-01]
# --------------------------------------------------------- Enter Experiment Info Above ---------------------------------------------------------

#Load some useful libraries
from ml_sl.adaboost.ab_0 import AB
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import itertools

#Takes Excel data and cuts off the first row, then returns the two column trial chosen 'x' 
def trimData(data, x):
    data =data[0:,x:x+2] 
    data = np.delete(data,(0),axis=0)
    out =(data-np.min(data))/(np.max(data)-np.min(data))
    return out

#Takes input array and list of index combinations, 'combs', runs an Adaboost prediction on each 
#combination of 'data' points. Creates a list of ECM predictions for each combination and returns a DataFrame
def method1(data,combs):  
    list1=[]    
    for i in combs:
        one,two,three,four,five = i[0],i[1],i[2],i[3],i[4]
        temp = np.concatenate((data[one,:],data[two,:],data[three,:],data[four,:],data[five,:]),axis=0)
        temp = np.reshape(temp,(5,2))
        # Use AdaBoost to predict an ECM
        # You can pick any number between 0 ~ 620 (total data amount of ml-dataset 629)
        # Runs a prediction on the unlabeled dataset.
        # Ada assigns higher weights to misclassified data to focus subsequent learners in the ensemble.
        ab = AB(boost_num=150, resample_num=10, alpha_init=1, max_iter=9000,
                        unlabeled_dataset_list=[temp],
                        labeled_dataset_list=tr_dataset,
                        label_list=label_list)
        abSampleLabelProbDictList = ab.classify(ab_model_name='./ml_sl/adaboost/models/trained_on_TV_tested_on_test/2020_07_06_ab_final_boost_num=150_3_pickle.file')
        list1.append(abSampleLabelProbDictList)
    temp = sum(list1,[])
    outPredictions = pd.DataFrame(temp)
    outPredictions.iloc[:,:] = Normalizer(norm='l1').fit_transform(outPredictions)
    outMax = outPredictions.idxmax()      
    return outPredictions,outMax                       

#Takes input data and runs an Adaboost prediction, removing a singular data
#point each time. Returns a list of prediction values for all 7 ECM models and the frequency that caused the 
#greatest drop in prediction accuracy for each model.
def method2(data,combs):
    list2=[]
    for i in range(np.size(data,0)):
        temp = np.delete(data,(i),axis=0)
        ab = AB(boost_num=150, resample_num=10, alpha_init=1, max_iter=9000,
                        unlabeled_dataset_list=[temp],
                        labeled_dataset_list=tr_dataset,
                        label_list=label_list)
        abSampleLabelProbDictList = ab.classify(ab_model_name='./ml_sl/adaboost/models/trained_on_TV_tested_on_test/2020_07_06_ab_final_boost_num=150_3_pickle.file')
        list2.append(abSampleLabelProbDictList)
    temp = sum(list2,[])
    outPredictions = pd.DataFrame(temp)
    outPredictions.iloc[:,:] = Normalizer(norm='l1').fit_transform(outPredictions)
    outMin = outPredictions.idxmin()
    return outPredictions,outMin
                                      
#Creates a list of all possible unique combinations of 'n' points in a given list length 'l'
def generateComb(n,l):
        out = []
        states = np.arange(0,n,1)
        for v in itertools.combinations(states, l):
             out.append(v)
        return out

# --------------------------------------------------------- ECM Data Combinations and Trimming  ---------------------------------------------------------

label_list = [2, 4, 5, 6, 7, 8, 9]
# Import sample ml-dataset (Training, validation, Test)
ml_dataset_pickle_file_path = './datasets/ml_datasets/normed'
tr_dataset, va_dataset, te_dataset = get_T_V_T_dataset(file_path=ml_dataset_pickle_file_path)
eisData = pd.read_excel(filePath)
eis = eisData.to_numpy()
eisClean = trimData(eis, trialNum)
ml_dataset = tr_dataset + va_dataset + te_dataset 
ml_label_list, ml_data_list = split_labeled_dataset_list(ml_dataset)

# --------------------------------------------------------- Running Both Analysis Methods  ---------------------------------------------------------

#dataCombs = generateComb(np.size(eisClean,0),5)
dataCombs = generateComb(8,5)
firstMethodData, tFF = method1(eisClean,dataCombs)
secondMethodData, mostImportantFreq =method2(eisClean)
topFiveTemp = np.array(tFF)
topFive = np.array([dataCombs[topFiveTemp[0]], dataCombs[topFiveTemp[1]], dataCombs[topFiveTemp[2]], 
           dataCombs[topFiveTemp[3]], dataCombs[topFiveTemp[4]], dataCombs[topFiveTemp[5]], 
           dataCombs[topFiveTemp[6]]])
topFreq = np.array(mostImportantFreq) 
glhf = dataCombs[0]

# --------------------------------------------------------- Displaying ECMs and Results  ---------------------------------------------------------

ecms = plt.imshow(mpimg.imread('ECModels.jpg'))
plt.axis('off')
print('ECM Model 2 Best Freqs: ' + str(eisFreqs[topFive[0,0]]) + "Hz "+ str(eisFreqs[topFive[0,1]]) + "Hz " + str(eisFreqs[topFive[0,2]]) + "Hz "+ str(eisFreqs[topFive[0,3]]) + "Hz "+ str(eisFreqs[topFive[0,4]]) + "Hz ")
print('ECM Model 2 Top Freq: ' + str(eisFreqs[topFreq[0]]) + "Hz" )
print('ECM Model 4 Best Freqs: ' + str(eisFreqs[topFive[1,0]]) + "Hz "+ str(eisFreqs[topFive[1,1]]) + "Hz " + str(eisFreqs[topFive[1,2]]) + "Hz "+ str(eisFreqs[topFive[1,3]]) + "Hz "+ str(eisFreqs[topFive[1,4]]) + "Hz ")
print('ECM Model 4 Top Freq: ' + str(eisFreqs[topFreq[1]]) + "Hz" )
print('ECM Model 5 Best Freqs: ' + str(eisFreqs[topFive[2,0]]) + "Hz "+ str(eisFreqs[topFive[2,1]]) + "Hz " + str(eisFreqs[topFive[2,2]]) + "Hz "+ str(eisFreqs[topFive[2,3]]) + "Hz "+ str(eisFreqs[topFive[2,4]]) + "Hz ")
print('ECM Model 5 Top Freq: ' + str(eisFreqs[topFreq[2]]) + "Hz" )
print('ECM Model 6 Best Freqs: ' + str(eisFreqs[topFive[3,0]]) + "Hz "+ str(eisFreqs[topFive[3,1]]) + "Hz " + str(eisFreqs[topFive[3,2]]) + "Hz "+ str(eisFreqs[topFive[3,3]]) + "Hz "+ str(eisFreqs[topFive[3,4]]) + "Hz ")
print('ECM Model 6 Top Freq: ' + str(eisFreqs[topFreq[3]]) + "Hz" )
print('ECM Model 7 Best Freqs: ' + str(eisFreqs[topFive[4,0]]) + "Hz "+ str(eisFreqs[topFive[4,1]]) + "Hz " + str(eisFreqs[topFive[4,2]]) + "Hz "+ str(eisFreqs[topFive[4,3]]) + "Hz "+ str(eisFreqs[topFive[4,4]]) + "Hz ")
print('ECM Model 7 Top Freq: ' + str(eisFreqs[topFreq[4]]) + "Hz" )
print('ECM Model 8 Best Freqs: ' + str(eisFreqs[topFive[5,0]]) + "Hz "+ str(eisFreqs[topFive[5,1]]) + "Hz " + str(eisFreqs[topFive[5,2]]) + "Hz "+ str(eisFreqs[topFive[5,3]]) + "Hz "+ str(eisFreqs[topFive[5,4]]) + "Hz ")
print('ECM Model 8 Top Freq: ' + str(eisFreqs[topFreq[5]]) + "Hz" )
print('ECM Model 9 Best Freqs: ' + str(eisFreqs[topFive[6,0]]) + "Hz "+ str(eisFreqs[topFive[6,1]]) + "Hz " + str(eisFreqs[topFive[6,2]]) + "Hz "+ str(eisFreqs[topFive[6,3]]) + "Hz "+ str(eisFreqs[topFive[6,4]]) + "Hz ")
print('ECM Model 9 Top Freq: ' + str(eisFreqs[topFreq[6]]) + "Hz" )