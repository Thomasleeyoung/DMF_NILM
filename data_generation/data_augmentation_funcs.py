# %% [markdown]
# **Data Augmentation Techniques**
# - Modifying mains frequency
# - Shifting in time(forwards and backwards)
# - Scaling(larger and smaller)


import scipy
import numpy as np
import matplotlib.pyplot as plt

#input: data, is an mxn array, with m examples, and n values for each timestamp
def change_mains_freq(old_freq, new_freq, data):
    num_samples = data.shape[1]
    if new_freq<old_freq: #if going to 60Hz from 50Hz
        new_num_samples = int((num_samples*old_freq)/new_freq)
        new_freq_data = scipy.signal.resample(data, new_num_samples, axis=1)[:,:num_samples]

    else: #if going from 50Hz to 60Hz
        downsample_ratio = round((new_freq/old_freq),1) #save to 1 decimal place(1.2)
        new_freq_data = scipy.signal.resample(data, num_samples*10, axis=1)
        #uses a downsample ratio of 12 on 10x data, is the same as an exact
        # downsample ratio of 1.2
        new_freq_data = new_freq_data[:,::int(downsample_ratio*10)]

    stacked_data = np.stack((data, new_freq_data), axis=-1) 
    #stacks onto last axis and creates new dimension, therefore the order that these functions
    #are carried out in does not matter
    return stacked_data


def window_shift(data, num_shifts, event_time=0.05, sample_freq=500):
    file_len = data.shape[1]
    #spacing between events between each shift
    rows_spacing = int((file_len-(event_time*sample_freq))/num_shifts)
    #repeat first values in array before event
    shift_arr = np.repeat((data[:, 0, ...])[:, np.newaxis, ...], (rows_spacing*num_shifts), axis=1)
    shift_arr = np.hstack((shift_arr, data))
    temp_list = []
    for i in range(num_shifts):
        rows_start = (num_shifts-i)*rows_spacing
        temp_arr = (shift_arr[:, rows_start:rows_start+file_len, ...])
        temp_list.append(temp_arr)
        #plt.plot(range(temp_arr.shape[1]), temp_arr[0, :, 0], label=i)
    shifted_data = np.stack(temp_list, axis = -1)
    return shifted_data


def scale(data, lower_multiplier = 0.8, upper_multiplier = 1.2, num_splits = 5):
    split_ints = np.linspace(lower_multiplier, upper_multiplier, num_splits)
    temp_list = []
    for i in split_ints:
        temp_arr = data * i
        temp_list.append(temp_arr)
    scaled_arr = np.stack(temp_list, axis=-1)
    return scaled_arr

