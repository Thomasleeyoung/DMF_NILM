import json
import pandas as pd
import numpy as np
import scipy


def read_plaid_meta():
    with open("c:/Users/tslee/IRP Data/PLAID_DATASET/meta_2014.json", "r") as readfile:
        data_2014=json.load(readfile)
    readfile.close()
    with open("c:/Users/tslee/IRP Data/PLAID_DATASET/meta_2017.json", "r") as readfile:
        data_2017=json.load(readfile)
    readfile.close()
    appliance_types=[]
    for val in data_2014:
        appliance_types.append([int(val["id"]),val["meta"]["appliance"]["type"]])
    for val in data_2017:
        appliance_types.append([int(val["id"]),val["meta"]["appliance"]["type"]])
        #created a list of lists of id and appliance types for all values
    #print(len(appliance_types))
    #meta_2014 ends on id=1074, and meta_2017 starts on id=1075
    types_df = pd.DataFrame(appliance_types, columns=("id", "appliance_type"))
    return types_df

def apparent_power(data_df, new_sample_freq, mains_freq):
    #data_df is the dataframe of 2 columns: current and voltage
    #sample_freq = 30000
    samples = int(new_sample_freq/mains_freq) #number of sample points per cycle

    #data_df.columns=["current", "voltage"]
    data_df["rms_current"] = (((abs(data_df["current"])**2).rolling(samples).mean())**0.5)
    data_df["rms_voltage"] = (((abs(data_df["voltage"])**2).rolling(samples).mean())**0.5)
    data_df["apparent_power"] = data_df["rms_current"]*data_df["rms_voltage"]
    data_df = data_df.dropna() #remove np.nan in beginning rows
    return data_df

def downsample(data_df, old_sample_rate, new_sample_rate, axis=0):
    step_size = np.round(old_sample_rate/new_sample_rate).astype("int")

    if axis==1: #downsample across columns
        downsampled_df = data_df.iloc[:, ::step_size]
    else: #downsample down rows by default
        downsampled_df = data_df.iloc[::step_size, :].reset_index() #format for each dimension is [start:stop:step]
    #therefore, this returns every step_size-th row, and all data from that row
    #resets the index and stores the old index as a column
    return downsampled_df

def event_start(data_df, threshold, num_rows_start, num_rows_end):
    dy = np.diff(data_df["apparent_power"])
    peakIndex, _ = scipy.signal.find_peaks(dy, height=threshold)
    if len(peakIndex)<1: #remove the example from dataset if there is no sufficiently large event
        #print("remove example from dataset")
        return data_df, True
    
    if peakIndex[0] <= num_rows_start: #if the device switches on before the allowable buffer time
        #end_index = num_rows_end
        if len(data_df["apparent_power"])>=num_rows_end: #if there as least num_rows_total in the file, then take the first num_rows_total rows
            return data_df.iloc[:num_rows_end], False
        else: #if the file is too short, it is a bad file
            return data_df, True
    else: #if peak occurs after allowable buffer
        if len(data_df["apparent_power"])-peakIndex[0]+num_rows_start>=num_rows_end: #if there are more than the min rows after the buffer start before the peak
            start_index = peakIndex[0]-num_rows_start
            end_index = start_index+num_rows_end
            return data_df.iloc[start_index:end_index].reset_index(), False
        else: #if not enough rows after start_buffer it is a bad file
            return data_df, True
