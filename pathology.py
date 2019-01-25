import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal

def derivative_clip(signal,t=0.1,plot_figure=True):
    """
    Performs the derivatve clipping signal processing algorithm

    Parameters:
    -----------
    signal: 1D numpy array with T samples
    t: derivative threshold
    plot_figure: whether to show plot objects

    Returns:
    --------
    ema_4: clipped derivative signal
    """

    t = 0.1

    ema_4_diff = np.diff(signal)
    ema_4_diff[ema_4_diff > t] = t
    ema_4_diff[ema_4_diff < -t] = -t
    ema_4 = np.cumsum(np.insert(ema_4_diff,0,signal[0]))

    if plot_figure:
        sns.set_style("darkgrid")
        plt.subplot(1,2,1)
        plt.plot(signal)
        plt.title("Original EMA signal for tongue tip")
        plt.ylim([-2, 3.5])
        plt.xlabel("time [s]")
        plt.ylabel("normalised x position")

        plt.subplot(1,2,2)
        plt.plot(ema_4)
        plt.title("Modified EMA signal for tongue tip")
        plt.ylim([-2, 3.5])
        plt.xlabel("time [s]")
        plt.ylabel("normalised x position")
        plt.show()

    return ema_4


def derivative_clip_2(dataset,t, channel_idx):

    path_test = dataset 
    path_test_diff = np.diff(dataset[:,:,channel_idx],axis=1)
    print(path_test_diff.shape)
    path_test_diff[:, path_test_diff > t, :] = t * np.ones_like(path_test_diff[:,path_test_diff > t, :])
    path_test_diff[:, path_test_diff < -t, :] = -t * np.ones_like(path_test_diff[:,path_test_diff < -t, :])
    path_test[:, :, channel_idx] = np.cumsum(np.insert(path_test_diff,0,path_test[:,0,:]))

    return path_test
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

def zero_channel(dataset,channel_num):
    path_test = dataset
    path_test[:,:,channel_num] = 0

    return path_test
