# File for preprocessing the MOCHA-TIMIT dataset
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns

import soundfile as sf
import sounddevice as sd

import time

from sklearn import preprocessing
from sklearn.externals import joblib
from scipy.signal import decimate,savgol_filter,resample_poly,resample
# Speech processing frameworks
import pyworld as pw
import python_speech_features as psf
import pysptk as sptk
from nnmnkwii.preprocessing import interp1d, modspec_smoothing, delta_features
from nnmnkwii.paramgen import mlpg

import tqdm as tqdm
import pandas as pd
from scipy.io import wavfile

import glob

        

def clean(s):
    """
    Strips the new line character from the buffer input
    Parameters:
    -----------
    s: Byte buffer

    Returns:
    --------
    p: string stripped from new-line character

    """
    s = str(s,"utf-8")
    return s.rstrip('\n').strip()

def core_read(fname,bias):
    """
    Reads in a single EMA file

    Parameters:
    -----------
    fname: Filename with extension .ema (String)
    bias: Implicit bias in shape (it is needed in one of the datasets)

    Returns:
    --------
    columns: columns name (i.e. which electrode channel) 
    data: 2D numpy array with sample points x channel
    """
    
    columns = {}
    columns["time"] = 0
    columns["present"] = 1

    with open(fname, 'rb') as f:

        line = f.readline() # EST File Track
        datatype = clean(f.readline()).split()[1]
        nframes = int(clean(f.readline()).split()[1])
        f.readline() # Byte Order
        nchannels = int(clean(f.readline()).split()[1])

        while not 'CommentChar' in str(f.readline(),"utf-8"):
            pass
        f.readline() # empty line
        line = clean(f.readline())

        while not "EST_Header_End" in line:
            channel_number = int(line.split()[0].split('_')[1]) + 2
            channel_name = line.split()[1]
            columns[channel_name] = channel_number
            line = clean(f.readline())

        string = f.read()
        data = np.fromstring(string, dtype='float32')
        data_ = np.reshape(data, (-1, len(columns) + bias ))

        return columns, data_

def ema_read(fname):
    """
    Reads in a single EMA file
    Aldo does the preprocessing based on the MNGU0 needs

    Parameters:
    -----------
    fname: Filename with extension .ema (String)

    Returns:
    --------
    a numpy array with the EMA data (time, channel)
    """

    columns, data = core_read(fname,0)

    # Fetch indexes for dereferencing
    px_index = [values for keys,values in columns.items() if "px" in keys]
    py_index = [values for keys,values in columns.items() if "py" in keys]
    pz_index = [values for keys,values in columns.items() if "pz" in keys]

    # Substract the reference values
    data[:,px_index] = data[:,px_index] - data[:,[columns["ref_px"]]]
    data[:,py_index] = data[:,py_index] - data[:,[columns["ref_py"]]]
    data[:,pz_index] = data[:,pz_index] - data[:,[columns["ref_pz"]]]

    # Select which keys to leave out
    no_keys =  ["ref","newflag","time","present","rms",
                "head","oz","px","ox","oy","taxdist"]

    keep_keys = [keys for keys,values in columns.items()]

    for keys,value in columns.items():
        for no_key in no_keys:
            if no_key in keys:
                try:
                    keep_keys.remove(keys)
                except ValueError:
                    pass

    keep_values = [columns[key] for key in keep_keys]

    data = data[:,keep_values]

    # Interpolate for nans
    for i in range(data.shape[1]):
        data[:,i] = pd.Series(data[:,i]).interpolate().values

    return data

  
def ema_read_mocha(fname):
    """
    Reads in a single EMA file
    Also does the preprocessing based on the MOCHA-TIMIT needs
    
    Parameters:
    -----------
    fname: Filename with extension .ema (String)

    Returns:
    --------
    a numpy array with the EMA data (time, channel)
    """

    co, data_ = core_read(fname,1)

    # Selecting channels which will be actually used in training
    idx = [co["td_x"],co["td_y"],co["tb_x"],co["tb_y"],co["tt_x"],
           co["tt_y"],co["li_x"],co["li_y"],co["ui_x"],co["ui_y"],
           co["ul_x"],co["ul_y"],co["ll_x"],co["ll_y"]]
    data = data_[:,idx]

    for k in range(data.shape[1]):
        data[:,k] = savgol_filter(data[:,k], 51, 3)

    return data

def debug_synth(f0,sp,ap,fs,an=2):
    """
    Plays a synthetised audio

    Parameters:
    -----------
    Parameters from the PyWORLD vocoder

    f0: fundamental frequency
    sp: spectrum
    ap: band aperiodcities
    fs: sampling frequency (typically 16000)
    an: length of the analysis window

    """
    
    sound = pw.synthesize(f0,sp,ap,fs, an)
    sd.play(sound,fs)
    sd.wait()

def debug_resynth(f0_,sp_,ap_,fs,an=2,alpha=0.42,fftbin=1024):
    """
    Plays a synthetised audio from the encoded parameters
    It is good to perform quick analysis-resynthesis

    Parameters:
    -----------
    Parameters from the PyWORLD vocoder

    f0: fundamental frequency
    sp: spectrum
    ap: band aperiodcities
    fs: sampling frequency (typically 16000)
    an: length of the analysis window
    alpha: pre-emphasis filtering coefficient
    fftbin: bin-size of the underlying FFT 
    """
    sp_ = sptk.conversion.mc2sp(sp_, alpha, fftbin)
    ap_ = pw.decode_aperiodicity(ap_, fs, fftbin)
    sound = pw.synthesize(f0_,sp_,ap_,fs,an)
    sd.play(sound*3,fs)
    sd.wait()

def save_resynth(fname,f0_,sp_,ap_,fs,an=2,alpha=0.42,fftbin=1024):
    """
    Plays a synthetised audio from the encoded parameters
    It is good to perform quick analysis-resynthesis
    It also saves the file in WAV format

    Parameters:
    -----------
    Parameters from the PyWORLD vocoder

    f0: fundamental frequency
    sp: spectrum
    ap: band aperiodcities
    fs: sampling frequency (typically 16000)
    an: length of the analysis window
    alpha: pre-emphasis filtering coefficient
    fftbin: bin-size of the underlying FFT 

    """
    sp_ = sptk.conversion.mc2sp(sp_, alpha, fftbin)
    ap_ = pw.decode_aperiodicity(ap_, fs, fftbin)
    sound = pw.synthesize(f0_,sp_,ap_,fs,an)
    sd.play(sound*3,fs)
    sd.wait()
    wavfile.write(fname,fs,sound*3)

def train_val_split(files,val_split):
    """
    Partitions the list into training and validation

    Parameters:
    -----------
    files: concatenated list with filenames
    val_split: percentage of validation split

    Returns:
    train_idx: training ids
    val_idx: validation ids
    """

    indices = np.arange(len(files))
    np.random.shuffle(indices)

    assert val_split < 1 and val_split > 0, \
        "Validation split must be a number on open interval (0,1)"

    validation_size = int(np.ceil(val_split * total_samples))
    val_idx = indices[:validation_size]
    train_idx = indices[validation_size:]

    return train_idx,val_idx

save_dir = "processed_comb2_filtered"

def preprocess_save_combined(normalisation=True,alpha=0.42,
                             max_length=1000, fs=16000, val_split=0.2,
                             noise=False,combined=False, bins_1 = 41,
                             bins_2 = 1, normalisation_input = True,
                             normalisation_output = True,
                             channel_number = 14,
                             factor = 80):
    """
    The main entry point to the preprocessing pipeline

    Parameters:
    -----------
    normalisation: whether to perform feature-wise normalisation
    alpha: pre-emphasis filtering
    max_length: maximum samples of speech and EMA. It is one to one
    because both is downsampled appropriately
    val_split: percentage validation split
    noise: whether to add noise to the data (generally a bad idea)
    combined: whether to use a single dataset (MNGU0) or the combined
    bins_1: MFCC channel numbers
    bins_2: BAP channel numbers
    normalisation_input: whether to normalise the EMA feature-wise
    normalisation_output: whether to normalise ths speech feature-wise
    channel_number: number of electrode channels in the files.
    factor: downsampling factor used    

    --------
    """

    # Fetching and shuffling the appropriate file lists
    if combined:
        files1 = np.array(sorted(glob.glob("dataset2/*.ema")))
        files2_m = np.array(sorted(glob.glob("dataset/msak*.ema")))
        files2_f = np.array(sorted(glob.glob("dataset/fsew*.ema")))
        files = np.concatenate((files1,files2_m,files2_f))
    else:
        files = np.array(glob.glob("dataset2/*.ema"))
    np.random.shuffle(files)
    total_samples = len(files)
    

    print("Preprocessing " + str(total_samples) + " samples")

    train_idx, val_idx = train_val_split(files,0.2)

    max_f0_length = max_length
    max_audio_length = max_length

    # Preallocation of memory
    dataset = np.zeros((total_samples,max_length,all_channel+1))
    puref0set = np.zeros((total_samples,max_f0_length))
    spset = np.zeros((total_samples,max_f0_length,bins_1 * 2))
    apset = np.zeros((total_samples,max_f0_length,bins_2))

    # Which ID correspond to which dataset, male, female 
    male_id = []
    female_id = []
    mngu0_id = []

    # Append the appropriate id and read files
    for k,fname in enumerate((files)):
        # Indicate progress
        print(k)
        if "mngu0" in fname:
            data_ = ema_read(fname)
            mngu0_id.append(k)
        else:
            data_ = ema_read_mocha(fname)
            if "fsew" in fname:
                female_id.append(k)
            if "msak" in fname:
                male_id.append(k)
            # Resample so the datasets have same sampling frequency
            data_ = resample(data_,int(np.ceil(data_.shape[0]*2/5)))

        # We don't need the time and present rows
        read_in_length = np.minimum(data_.shape[0],max_length)
        dataset[k,:read_in_length,:-1] = data_[:read_in_length,:]

        # Repeating last elements
        if (max_length > data_.shape[0]):
            dataset[k,data_.shape[0]:,:-1] = data_[data_.shape[0]-1,:]
        if np.isnan(dataset).any():
            dataset[np.isnan(dataset)] = 0
            print("Warning! Zeroed a NaN")

        fname_wo_extension = fname[:-3 or None]
        wav_path = fname_wo_extension + "wav"
        sound_data, fs = sf.read(wav_path)

        f0, sp, ap = pw.wav2world(sound_data, fs, 5) # 2

        # The general way is to either truncate or zero-pad
        read_in_length = np.minimum(f0.shape[0],max_f0_length)
        dataset[k,:read_in_length,all_channel] = f0[:read_in_length]

        # Because of the 0th order spectra is needed, we use -1 for bin size
        sp = sptk.conversion.sp2mc(sp, bins_1 - 1, alpha)

        # TODO: Assumed here that the analysis window is the hop length
        hop_length = 20
        s_sp = modspec_smoothing(sp,200)

        # Getting the delta features
        windows = [
            (0, 0, np.array([1.0])),            # static
            (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
        ]
        sp_delta = delta_features(s_sp,windows)

        # Encode the spectral envelopes
        ap = pw.code_aperiodicity(ap, fs)

        # DEBUG: Decode spectral envelope
        dummy_var = 0.001*np.ones((sp_delta.shape[0],sp_delta.shape[1]))
        sp_dec = mlpg(sp_delta,dummy_var,windows)

        # Pure f0set
        read_in_length = np.minimum(f0.shape[0],max_f0_length)
        puref0set[k,0:read_in_length] = f0[0:read_in_length]

        # Spectrum
        read_in_length = np.minimum(sp.shape[0],max_f0_length)
        spset[k,0:read_in_length,:] = sp_delta[0:read_in_length,:]

        # Band aperiodicites
        read_in_length = np.minimum(ap.shape[0],max_f0_length)
        apset[k,0:read_in_length,:] = ap[0:read_in_length,:]


    if normalisation_input:

        # Normalise the articulographs differently for different references
        train_male = list(set(train_idx).intersection(male_id))
        train_female = list(set(train_idx).intersection(female_id))
        train_mngu0 = list(set(train_idx).intersection(mngu0_id))
        val_male = list(set(val_idx).intersection(male_id))
        val_female = list(set(val_idx).intersection(female_id))
        val_mngu0 = list(set(val_idx).intersection(mngu0_id))

        # Normalise ema feature wise but do not return normaliser object
        for j in range(all_channel):
            scaler_ema1 = preprocessing.StandardScaler()
            scaler_ema2 = preprocessing.StandardScaler()
            scaler_ema3 = preprocessing.StandardScaler()
            dataset[train_male,:,j] = scaler_ema1.fit_transform(dataset[train_male,:,j])
            dataset[train_female,:,j] = scaler_ema2.fit_transform(dataset[train_female,:,j])
            dataset[train_mngu0,:,j] = scaler_ema3.fit_transform(dataset[train_mngu0,:,j])
            dataset[val_male,:,j] = scaler_ema1.transform(dataset[val_male,:,j])
            dataset[val_female,:,j] = scaler_ema2.transform(dataset[val_female,:,j])
            dataset[val_mngu0,:,j] = scaler_ema3.transform(dataset[val_mngu0,:,j])

    if normalisation_output:
        # Spectrum scalers
        scaler_sp = []
        for k in range(bins_1):
            scaler_sp.append(preprocessing.StandardScaler())
            spset[train_idx,:,k] = scaler_sp[k].fit_transform(spset[train_idx,:,k])
            spset[val_idx,:,k] = scaler_sp[k].transform(spset[val_idx,:,k])

        # Aperiodicities scalers
        scaler_ap = []
        for k in range(bins_2):
            scaler_ap.append(preprocessing.StandardScaler())
            apset[train_idx,:,k] = scaler_ap[k].fit_transform(apset[train_idx,:,k])
            apset[val_idx,:,k] = scaler_ap[k].fit_transform(apset[val_idx,:,k])

    np.save(save_dir + "/dataset_", dataset)
    np.save(save_dir + "/puref0set_", puref0set)
    np.save(save_dir + "/spset_", spset)
    np.save(save_dir + "/apset_", apset)
    np.save(save_dir + "/train_idx_", train_idx)
    np.save(save_dir + "/val_idx_", val_idx)

    joblib.dump(scaler_sp, save_dir + '/scaler_sp_.pkl')
    joblib.dump(scaler_ap, save_dir + '/scaler_ap_.pkl')


def delay_signal(signal,delay):
    """
    Wrapper for numpy boilerplate for delaying the signal.
    The idea is that signal can be delay by shifting and zero padding
    in the beginning.
    

    Parameters:
    -----------
    signal: The signals are assumed to be tensors of either (Sample X 
    Time x Channel) or (Sample X Time)

    Returns: 
    delayed_signal: the delayed signal same shape as signal
    """

    assert len(signal.shape) < 4, "Invalid signal shape"
    assert len(signal.shape) < 1, "Invalid signal shape"
    if len(signal.shape) == 2:
        delayed_signal = np.pad(signal,
                                ((0,0),(delay,0)),
                                mode="constant")[:,:-delay]
    else:
        delayed_signal = np.pad(signal,
                                ((0,0),(delay,0),(0,0)),
                                mode="constant")[:,:-delay,:]
    return delayed_signal

def load_test(delay,percentage=1):
    """
    Loads the dataset for testing.
    
    Parameters:
    -----------
    delay: number of delay to use
    percentage: percentage of the dataset to load. TODO: Does not
    really make sense for testing, should be deprecated.
    
    Returns:
    --------
    The test sets as numpy arrays and the scaler objects
    """
    
    dataset = np.load(save_dir + "/dataset_.npy")
    spset = np.load(save_dir + "/spset_.npy")
    apset = np.load(save_dir + "/apset_.npy")
    puref0set = np.load(save_dir + "/puref0set_.npy")
    scaler_sp = joblib.load(save_dir + '/scaler_sp_.pkl')
    scaler_ap = joblib.load(save_dir + '/scaler_ap_.pkl')
    train_idx = np.load(save_dir + "/train_idx_.npy")
    test_idx = np.load(save_dir + "/val_idx_.npy")

    # Reduce training id size. It is shuffled by default so it is not reshuffled for brevity
    keep_amount = int(np.ceil(percentage * len(train_idx)))
    train_idx = train_idx[:keep_amount]
    
    ema_test = dataset[test_idx,:,:]
    puref0_test = delay_signal(puref0set[test_idx,:],delay)
    sp_test = delay_signal(spset[test_idx,:,:],delay)
    ap_test = delay_signal(apset[test_idx,:,:],delay)

    return ema_test, sp_test, ap_test,puref0_test, None, scaler_sp, \
        scaler_ap
def load_data(delay,percentage=1):
    """
    Loads the dataset for testing AND training.
    
    Parameters:
    -----------
    delay: number of delay to use
    percentage: percentage of the dataset to load. 
    
    Returns:
    --------
    The test sets as numpy arrays and the scaler objects
    """
    
    dataset = np.load(save_dir + "/dataset_.npy")
    spset = np.load(save_dir + "/spset_.npy")
    apset = np.load(save_dir + "/apset_.npy")
    scaler_sp = joblib.load(save_dir + '/scaler_sp_.pkl')
    train_idx = np.load(save_dir + "/train_idx_.npy")
    test_idx = np.load(save_dir + "/val_idx_.npy")

    # Reduce training id size. It is shuffled by default so it is not reshuffled for brevity
    keep_amount = int(np.ceil(percentage * len(train_idx)))
    train_idx = train_idx[:keep_amount]
    
    # EMA and LAR partition
    ema_train = dataset[train_idx,:,:]
    ema_test = dataset[test_idx,:,:]
    dataset = None
    # Padding f0
    f0_train = None
    f0_test = None
    f0set = None
    
    sp_train = delay_signal(spset[train_idx,:,:],delay)
    sp_test = delay_signal(spset[test_idx,:,:],delay)
    spset = None

    ap_train = delay_signal(apset[train_idx,:,:],delay)
    ap_test = delay_signal(apset[test_idx,:,:],delay)
    apset = None

    givenf0_train = None
    givenf0_test = None
    givenf0set = None

    return ema_train, ema_test, \
        None, None, \
        sp_train, sp_test, \
        None, None, \
        None, None, \
        None, None, scaler_sp, None

#preprocess_save_combined(normalisation=True,alpha=0.42,max_length=1000,
#                                                                  fs=16000, val_split=0.1,
#                         noise=False,combined=True)

