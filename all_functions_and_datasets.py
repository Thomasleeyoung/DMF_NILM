# %%
from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import pandas as pd
from glob import glob
import re
import scipy
import os
import sklearn
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer
import json
import pickle
import keras
import tensorflow as tf
from keras.models import load_model
import model_functions as mf
import argparse
import itertools
import time

# %%
#getting current time
start_time = time.time()

# %% [markdown]
# #defining the parser to read arguments from command line
parser = argparse.ArgumentParser("Arg parser")
parser.add_argument("--idx") 
args = vars(parser.parse_args())

job_idx = int(args["idx"])

# %%
#gets all possible combinations of changeable parameters
###COMMENT THIS NEXT LINE OUT
#job_idx = 11 ##################################
#################################

algorithms = ["benchmark", "NN"]
datasets = ["plaid", "whited", "both"]
input_sources = ["AC", "RMS"]
combinations = list(itertools.product(algorithms, datasets, input_sources))

job_params = combinations[job_idx]
algorithm = job_params[0]
dataset = job_params[1]
input_source = job_params[2]

# %%
#important parameters
test_samples = 5000 #number of test examples
original_freq = 5000
sample_size_range = [50, 100, 500, 1000] #range of training samples per class
#save_path = "C:/Users/tslee/IRP Data/DMF_Internship/DATA/"
#general_report_path = "C:/Users/tslee/IRP Data/DMF_Internship/DATA/"

save_path = "/user/work/iw20981/DMF_work/DATA/"
general_report_path = "/user/work/iw20981/DMF_work/third_run/"

#assigning relevant input type
#probably don't need this, just have the input files labelled accordingly
if input_source=="AC":
    freq_range = [150, 500, 1000] #range of freqs in Hz
    base_report_path = general_report_path + "AC_"
elif input_source=="RMS":
    freq_range = [10, 50, 150] #range of freqs in Hz
    base_report_path = general_report_path + "RMS_"

#reshaping function
def arr_reshape(data, labels):
    rows = data.shape[0]
    for i in data.shape[2:]:
        rows *= i
    tpose_tuple = list(np.arange(data.ndim))[::-1][:-2]
    tpose_tuple.extend([0, 1])
    tpose_tuple = tuple(tpose_tuple)
    data_reshaped = data.transpose(tpose_tuple).reshape((rows, data.shape[1]))
    labels_duplicated = np.reshape(np.tile(labels, int(rows/data.shape[0])), (-1, 1))
    return data_reshaped, labels_duplicated
#classes
whited_classes = np.array(['AC', 'CFL', 'Fan', 'Fridge', 'HairDryer',
                            'Heater', 'LightBulb', 'Laptop', 'Microwave', 'VacuumCleaner',
                            'WashingMachine'])
plaid_classes = np.array(['Air Conditioner', "Compact Fluorescent Lamp", 'Fan', 'Fridge', 'Hairdryer',
                            'Heater', "Incandescent Light Bulb", 'Laptop', 'Microwave', 'Vacuum',
                            'Washing Machine'])

# %%
#loading in relevant dataset
if dataset=="plaid":
    loaded_data = np.load((save_path + "PLAID_5000Hz_"+ input_source +"_AUGMENTED_data.npy"), allow_pickle=True)
    loaded_labels = np.load((save_path + "PLAID_5000Hz_labels.npy"), allow_pickle=True).reshape((-1,))
    #flattening the data into 2D, along with duplicating corresponding labels
    data_reshaped, labels_duplicated = arr_reshape(loaded_data, loaded_labels)
    del(loaded_data)
    #assigning report_path
    report_path = base_report_path + "plaid_reports/"
    pass

elif dataset=="whited":
    loaded_data = np.load((save_path + "WHITED_5000Hz_"+ input_source +"_AUGMENTED_data.npy"), allow_pickle=True)
    loaded_labels = np.load((save_path + "WHITED_5000Hz_"+ input_source +"_labels.npy"), 
                            allow_pickle=True).reshape((-1,)).astype("<U30")

    whited_mask = np.isin(loaded_labels, whited_classes).reshape((-1,))
    loaded_labels = loaded_labels[whited_mask]
    loaded_data = loaded_data[whited_mask]
    
    for idx, class_label in enumerate(whited_classes):
        loaded_labels[loaded_labels == class_label] = plaid_classes[idx]
        whited_rows = loaded_data.shape[0]

    data_reshaped, labels_duplicated = arr_reshape(loaded_data, loaded_labels)
    del(loaded_data)
    #assigning report_path
    report_path = base_report_path + "whited_reports/"
    pass

elif dataset=="both":
    plaid_data = np.load((save_path + "PLAID_5000Hz_"+ input_source +"_AUGMENTED_data.npy"), allow_pickle=True)
    plaid_labels = np.load((save_path + "PLAID_5000Hz_labels.npy"), allow_pickle=True).reshape((-1,))
    whited_data = np.load((save_path + "WHITED_5000Hz_"+ input_source +"_AUGMENTED_data.npy"), allow_pickle=True)
    whited_labels = np.load((save_path + "WHITED_5000Hz_"+ input_source +"_labels.npy"), 
                            allow_pickle=True).reshape((-1,)).astype("<U30")

    #only using whited data whose labels match and changing labels to match plaid
    whited_mask = np.isin(whited_labels, whited_classes).reshape((-1,))
    clean_whited_data = whited_data[whited_mask]
    clean_whited_labels = whited_labels[whited_mask]
    for idx, class_label in enumerate(whited_classes):
        clean_whited_labels[clean_whited_labels == class_label] = plaid_classes[idx]
    del(whited_data)

    #Unravelling 5D data arrs to 2D, along with duplicating labels accordingly
    plaid_reshaped, plaid_labels_duplicated = arr_reshape(plaid_data, plaid_labels)
    del(plaid_data)
    whited_reshaped, whited_labels_duplicated = arr_reshape(clean_whited_data, clean_whited_labels)
    del(clean_whited_data)

    #combining PLAID and WHITED data into one array
    data_reshaped = np.vstack((plaid_reshaped, whited_reshaped))
    del(plaid_reshaped, whited_reshaped)
    labels_duplicated = np.vstack((plaid_labels_duplicated, whited_labels_duplicated))
    del(plaid_labels_duplicated, whited_labels_duplicated)

    #assigning report_path
    report_path = base_report_path + "plaid_and_whited_reports/"
    pass

# %%
#standard preprocessing step applied to all data
#normalising the data
def norm_func(row):
    return (row-row.mean())/row.std()
norm_data = np.apply_along_axis(norm_func, 1, data_reshaped)
del(data_reshaped)

#getting random indices for the test set
test_indices = np.random.choice(norm_data.shape[0], test_samples, replace=False)
#indices not in test set
X_remaining_indices = np.delete(np.arange(norm_data.shape[0]), test_indices)

# %%
#one hot encoding the labels as required by Xgboost and NNs
def label_conversion(labels):
    Y_labels = LabelBinarizer()
    Y_labels.fit(labels)
    label_classes = Y_labels.classes_
    Y_labels = Y_labels.transform(labels)
    return Y_labels, label_classes
#get number of expected labels
binarizer = LabelBinarizer()
binarizer.fit(labels_duplicated)
label_classes = binarizer.classes_
Y_onehot = binarizer.transform(labels_duplicated)

# %%
#carrying out relevant algorithm
if algorithm=="NN":
    #creating an error log file if it doesn't exist already
    #if "NN_log.txt" not in glob((report_path + "*")):
    with open((report_path + "NN_log.txt"), "w+") as err_file:
        err_file.write("Tests: \n")

    for freq in freq_range:
        for sample_size in sample_size_range:
            """#get indices for the correct number of training samples from each class
            indices_list = []
            for label in label_classes:
                #indices of available instances from each class
                label_indices = np.nonzero(labels_duplicated[X_remaining_indices] == label)[0]
                #get certain number of samples from each class
                label_indices = list(np.random.choice(label_indices, sample_size))
                indices_list.extend(label_indices)
            train_indices = np.asarray(indices_list)"""
            
            #Method to take n total samples which are randomly selected from total sample set
            total_samples = sample_size*len(label_classes)
            train_indices = np.random.choice(X_remaining_indices, total_samples)

            #gets the indices of the current training set, ensuring not to use the same indices as the test set
            downsample_factor = int(original_freq/freq)
            rounded_rows = downsample_factor*freq #in case the new frequency is not exactly divisible
            X_train = norm_data[train_indices, :rounded_rows]
            X_train = np.mean(np.reshape(X_train, (X_train.shape[0], -1, downsample_factor)), axis=2)
            
            #onehot encode labels
            Y_onehot, label_classes = label_conversion(labels_duplicated)
            #Y_train = np.reshape(labels_duplicated[train_indices], (-1, ))
            Y_train_onehot = Y_onehot[train_indices]

            X_test = np.mean(np.reshape(norm_data[test_indices, :rounded_rows], (test_indices.shape[0], -1, downsample_factor)), axis=2)
            
            Y_test = labels_duplicated[test_indices, 0]
            Y_test_onehot = Y_onehot[test_indices, :]
            
            #print("Using " + str(sample_size) + " training samples," + " at " + str(freq) + " Hz")
            #try to execute NN function, or add to error log file
            try: #Neural Networks
                test_time = time.time()
                mf.NN_func(X_train, X_test, Y_train_onehot, Y_test_onehot, label_classes, 
                            report_path, num_epochs=1500)
                with open((report_path + "NN_log.txt"), "a+") as log_file:
                     log_file.write("NN " + str(sample_size) + " sample size " + "at " + str(freq) + " Hz :" 
                                    + str(time.time()-test_time) + " seconds." + "\n")
                     log_file.write("_" *10+ "\n")
                print("NN Saved")
            except Exception as error:
                with open((report_path + "NN_log.txt"), "a+") as err_file:
                    err_file.write("NN " + str(sample_size) + " sample size " + "at " + str(freq) + " Hz \n")
                    err_file.write("AN ERROR OCCURED: " + str(error) + "\n")
                    err_file.write("_" *10+ "\n")
            #break
        #break
    with open((report_path + "NN_log.txt"), "a+") as err_file:
                err_file.write("Total Run Time: " + str(time.time() - start_time) + " seconds")

elif algorithm=="benchmark":
    #creating an error log file if it doesn't exist already
    #if "benchmark_log.txt" not in glob((report_path + "*")):
    with open((report_path + "benchmark_log.txt"), "w+") as err_file:
        err_file.write("Tests: \n")

    for freq in freq_range:
        for sample_size in sample_size_range:
            """#get indices for the correct number of training samples from each class
            indices_list = []
            for label in label_classes:
                #indices of available instances from each class
                label_indices = np.nonzero(labels_duplicated[X_remaining_indices] == label)[0]
                #get certain number of samples from each class
                label_indices = list(np.random.choice(label_indices, sample_size))
                indices_list.extend(label_indices)
            train_indices = np.asarray(indices_list)"""

            #Method to take n total samples which are randomly selected from total sample set
            total_samples = sample_size*len(label_classes)
            train_indices = np.random.choice(X_remaining_indices, total_samples)
            
            #gets the indices of the current training set, ensuring not to use the same indices as the test set
            downsample_factor = int(original_freq/freq)
            rounded_rows = freq*downsample_factor
            X_train = norm_data[train_indices, :]
            X_train = np.mean(np.reshape(X_train[:, :rounded_rows], (X_train.shape[0], -1, downsample_factor)), axis=2)
            
            Y_train = np.reshape(labels_duplicated[train_indices], (-1, ))
            Y_train_onehot = Y_onehot[train_indices, :]

            X_test = np.mean(np.reshape(norm_data[test_indices, :rounded_rows], (test_indices.shape[0], -1, downsample_factor)), axis=2)
            
            #onehot encode labels
            Y_onehot, label_classes = label_conversion(labels_duplicated)
            Y_test = labels_duplicated[test_indices, 0]
            Y_test_onehot = Y_onehot[test_indices, :]
            
            #print("Using " + str(sample_size) + " training samples," + " at " + str(freq) + " Hz")
            #try to execute each function, or add to error log file
            try: #KNN
                test_time = time.time()
                mf.knn_func(X_train, X_test, Y_train, Y_test, report_path)
                with open((report_path + "benchmark_log.txt"), "a+") as log_file:
                     log_file.write("KNN " + str(sample_size) + " sample size " + "at " + str(freq) + " Hz :" 
                                    + str(time.time()-test_time) + " seconds." + "\n")
                     log_file.write("_" *10+ "\n")
                print("KNN Saved")
            except Exception as error:
                with open((report_path + "benchmark_log.txt"), "a+") as err_file:
                    err_file.write("KNN " + str(sample_size) + " sample size " + "at " + str(freq) + " Hz \n")
                    err_file.write("AN ERROR OCCURED: " + str(error) + "\n")
                    err_file.write("_" *10+ "\n")

            try: #SVM
                test_time = time.time()
                mf.svm_func(X_train, X_test, Y_train, Y_train_onehot, Y_test, report_path)
                with open((report_path + "benchmark_log.txt"), "a+") as log_file:
                     log_file.write("SVM " + str(sample_size) + " sample size " + "at " + str(freq) + " Hz :" 
                                    + str(time.time()-test_time) + " seconds." + "\n")
                     log_file.write("_" *10+ "\n")
                print("SVM Saved")
            except Exception as error:
                with open((report_path + "benchmark_log.txt"), "a+") as err_file:
                    err_file.write("SVM " + str(sample_size) + " sample size " + "at " + str(freq) + " Hz \n")
                    err_file.write("AN ERROR OCCURED: " + str(error) + "\n")
                    err_file.write("_" *10 + "\n")

            try: #XGBOOST
                test_time = time.time()
                mf.xgb_func(X_train, X_test, Y_train_onehot, Y_test_onehot, label_classes, report_path)
                with open((report_path + "benchmark_log.txt"), "a+") as log_file:
                     log_file.write("XGBoost " + str(sample_size) + " sample size " + "at " + str(freq) + " Hz :" 
                                    + str(time.time()-test_time) + " seconds." + "\n")
                     log_file.write("_" *10+ "\n")
                print("XGBoost Saved")
            except Exception as error:
                with open((report_path + "benchmark_log.txt"), "a+") as err_file:
                    err_file.write("XGBoost " + str(sample_size) + " sample size " + "at " + str(freq) + " Hz \n")
                    err_file.write("AN ERROR OCCURED: " + str(error) + "\n")
                    err_file.write("_" *10+ "\n")

            #break
        #break
    with open((report_path + "benchmark_log.txt"), "a+") as err_file:
                err_file.write("Total Run Time: " + str(time.time() - start_time) + " seconds")
