import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal


def derivative_clip_full(signal,t,channel_idx):
    """
    Performs the derivatve clipping signal processing algorithm

    Parameters:
    -----------
    signal: full 3D array
    channel_idx: channel ids to select


    Returns:
    --------
    new 3D array
    """

    ema_signal = np.copy(signal)

    plt.subplot(1,2,1)
    plt.plot(ema_signal[10,:,channel_idx].T)
    
    # Time derivative
    ema_diff = np.diff(ema_signal[:,:,channel_idx],axis=1)
    ema_initial = ema_signal[:,[[0]],channel_idx]

    ema_diff[ema_diff > t] = t
    ema_diff[ema_diff < -t] = -t

    ema_diff = np.concatenate((ema_initial, ema_diff), axis=1)
    ema_path_dims = np.cumsum(ema_diff, axis=1)

    ema_path = np.copy(ema_signal)
    ema_path[:,:,channel_idx] = ema_path_dims

    plt.subplot(1,2,2)
    plt.plot(ema_path[10,:,channel_idx].T)
    plt.show()
    return ema_path


def upsampling(dataset, total = 6500, channels=[4,9]):
    """
    Does the upsampling on the entire dataset

    Parameters:
    -----------
    dataset - sampled x time x channels numpy array
    t - total samples of the upsampled signal


    Returns:
    -----------
    path_test - the pathological articulatory sigal
    """


    path_test = np.copy(dataset)
    T = dataset.shape[1]
    
    for channel in channels:
        path_test[:,:T,channel] = signal.resample(dataset[:,:,channel],
                                                     total,
                                                     axis=1)[:,:T]

    return path_test


def zero_interval(dataset,begin,end):
    """
    Zeroes an interval in the speech to mute
    """
    
    path_test = dataset
    path_test[:,begin:end,:] = 0
    
    return path_test

def set_channel(dataset,value,channel_num):
    path_test = np.copy(dataset)
    path_test[:,:,channel_num] = value

    return path_test

def scale_channel(dataset,value,channel_num):
    path_test = np.copy(dataset)
    path_test[:,:,channel_num] = path_test[:,:,channel_num] / value

    plt.subplot(1,2,1)
    plt.plot(dataset[10,:,4])
    plt.subplot(1,2,2)
    plt.plot(path_test[10,:,4])
    plt.show()
    
    return path_test

def add_noise(dataset,noise_val,channel_num):
    path_test = np.copy(dataset)

    print(path_test[:,:,channel_num].shape)

    print(path_test[10,100,channel_num])
    path_test[:,:,channel_num] = path_test[:,:,channel_num] + np.random.normal(0,noise_val)
    print(path_test[10,100,channel_num])
    

    plt.subplot(1,2,1)
    plt.plot(dataset[10,:,5])
    plt.subplot(1,2,2)
    plt.plot(path_test[10,:,5])
    plt.show()
    
    return path_test

def delay_signal(signal,delay,channel_idx):
    """
    Wrapper for numpy boilerplate for delaying the signal.
    The idea is that signal can be delay by shifting and zero padding
    in the beginning.
    

    Parameters:
    -----------
    signal: The signal isassumed to be tensors of either
    - 3D (Sample X Time x Channel)

    Returns:
    --------
    delayed_signal: the delayed signal same shape as signal
    """

    assert len(signal.shape) == 3, "Invalid signal shape"
    assert delay >= 0, "Only positive delay could be applied"

    if (delay == 0):
        return signal

    delayed_signal = np.copy(signal)
    delayed_signal[:,:,channel_idx] = np.pad(signal[:,:,channel_idx],
                            ((0,0),(delay,0),(0,0)),
                            mode="constant")[:,:-delay,:]

    return delayed_signal
