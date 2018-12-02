
# File for preprocessing the MOCHA-TIMIT dataset
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


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
from nnmnkwii.preprocessing import interp1d

import tqdm as tqdm
import pandas as pd
from scipy.io import wavfile

import glob

        

# Strips the new line character from the buffer input
def clean(s):
    # Byte -> String
    s = str(s,"utf-8")
    return s.rstrip('\n').strip()

# Handles the .ema files
def ema_read(fname):
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
        thing = len(columns)
        data_ = np.reshape(data, (-1, thing))
        data = data_

        #print(columns)
        # Fetch indexes for dereferencing
        px_index = [values for keys,values in columns.items() if "px" in keys]
        py_index = [values for keys,values in columns.items() if "py" in keys]
        pz_index = [values for keys,values in columns.items() if "pz" in keys]

        # Substract the reference values
        data[:,px_index] = data[:,px_index] - data[:,[columns["ref_px"]]]
        data[:,py_index] = data[:,py_index] - data[:,[columns["ref_py"]]]
        data[:,pz_index] = data[:,pz_index] - data[:,[columns["ref_pz"]]]

        # Select the relevant keys
        no_keys =  ["ref","newflag","time","present","rms","head","oz","px","ox","oy","taxdist"]
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


        for i in range(data.shape[1]):
            data[:,i] = pd.Series(data[:,i]).interpolate().values

        return data

  
def ema_read_mocha(fname):
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
            channel_number = int(line.split()[0].split('_')[1])
            channel_name = line.split()[1]
            columns[channel_name] = channel_number + 2
            line = clean(f.readline())
        string = f.read()
        data = np.fromstring(string, dtype='float32')
        data_ = np.reshape(data, (-1, len(columns) + 1 ))

        # Filtering
        # Lower Incisor -> Jaw
        # Upper Incisor -> Nose
        # Tongue Dorsum Tongue Blade Tongue Tip
        # T3 (13,18) T2 (12,16) T1 
        # T3 (12,18) T2 (11,17) T1 (5,10) LI (3,8) UI (2,6) UL(4,9) LL(5,10)
        co = columns
        idx = [co["td_x"],co["td_y"],co["tb_x"],co["tb_y"],co["tt_x"],
               co["tt_y"],co["li_x"],co["li_y"],co["ui_x"],co["ui_y"],
               co["ul_x"],co["ul_y"],co["ll_x"],co["ll_y"]]

        
        data = data_[:,idx]
        for k in range(data.shape[1]):
            data[:,k] = savgol_filter(data[:,k], 51, 3)
            
        
        return data

def data_combine():
    files = np.array(glob.glob("dataset/*.ema"))
    files2 = np.array(glob.glob("dataset2/*.ema"))
    points1 = np.zeros((len(files),14))
    points2 = np.zeros((len(files2),14))

    for idx,fname in enumerate(files):
        data_ = ema_read_mocha(fname)
        points1[idx,:] = data_[0,:]
    
    for idx,fname in enumerate(files2):
        data__ = ema_read(fname)
        points2[idx,:] = data__[0,:]

    points1_mean = np.mean(points1,axis=0)
    points2_mean = np.mean(points2,axis=0)
    scale = np.std((points2 - points2_mean),axis=0)/np.std((points1 - points1_mean),axis=0)
    return scale,points1_mean,points2_mean
    
def debug_synth(f0,sp,ap,fs,an=2):
    sound = pw.synthesize(f0,sp,ap,fs,an)
    sd.play(sound,fs)
    sd.wait()

def debug_resynth(f0_,sp_,ap_,fs,an=2,alpha=0.42,fftbin=1024):
    sp_ = sptk.conversion.mc2sp(sp_, alpha, fftbin)
    ap_ = pw.decode_aperiodicity(ap_, fs, fftbin)
    sound = pw.synthesize(f0_,sp_,ap_,fs,an)
    sd.play(sound,fs)
    sd.wait()
    
def preprocess_save_combined(normalisation=True,alpha=0.42,
                             max_length=1000, fs=16000, val_split=0.2,
                    noise=False):

    files1 = np.array(glob.glob("dataset2/*.ema"))
    files2 = np.array(glob.glob("dataset/*.ema"))
    files = np.concatenate((files1,files2))
    
    np.random.shuffle(files)

    total_samples = len(files)

    print("Preprocessing " + str(total_samples) + " samples")

    # Partion the list into training and validation file lists to avoid memory overhead
    indices = np.arange(len(files))
    np.random.shuffle(indices)

    assert val_split < 1 and val_split > 0, \
        "Validation split must be a number on open interval (0,1)"

    validation_size = int(np.ceil(val_split * total_samples))
    val_idx = indices[:validation_size]
    train_idx = indices[validation_size:]

    # Some built-in parameters
    all_channel = 14
    max_f0_length = max_length
    factor = 80
    # Audio is down sampled appropriately by the analysis window
    max_audio_length = max_length
    bins_1 = 41
    bins_2 = 1
    
    dataset = np.zeros((total_samples,max_length,all_channel))
    puref0set = np.zeros((total_samples,max_f0_length))
    spset = np.zeros((total_samples,max_f0_length,bins_1))
    apset = np.zeros((total_samples,max_f0_length,bins_2))
    wavdata = np.zeros((total_samples,max_audio_length))
    scale,mean1,mean2 = data_combine()
    
    # Shuffling train_test ids
    for k,fname in enumerate((files)):
        print(k)
        if "mngu0" in fname:
 #           print(fname)
#            print("mngu0")
            data_ = ema_read(fname)
        else:
  #          print(fname)
            data_ = ema_read_mocha(fname)
            data_ = (data_ - mean1)*scale + mean2
            data_ = resample(data_,int(np.ceil(data_.shape[0]*2/5)))
            #data_ = resample_poly(data_,2,5)

        # We dont need the time and present rows
        read_in_length = np.minimum(data_.shape[0],max_length)
        dataset[k,:read_in_length,:] = data_[:read_in_length,:]
        if (max_length > data_.shape[0]):
            dataset[k,data_.shape[0]:,:] = data_[data_.shape[0]-1,:]
        if np.isnan(dataset).any():
         #   print("Found NaN! Showing plot..")
        #    plt.plot(dataset[k,:])
        #    plt.show()
            dataset[np.isnan(dataset)] = 0
            print("Warning! Zeroed a NaN")

        # Read wav
        wav_path = fname[:-3 or None] + "wav"
        sound_data, fs = sf.read(wav_path)

        # DEBUG: SD play check
        #sd.play(sound_data,fs)
        #sd.wait()
        # Read in either to max lenth (truncation) or when data is available (zero padding)

        read_in_length = np.minimum(sound_data.shape[0],80*max_audio_length)
        #wav_file = np.zeros((max_audio_length))
        #wav_file[0:read_in_length] = sound_data[0:read_in_length]
        #if noise:
        #    wav_file = wav_file + np.random.normal(0,0.003,wav_file.shape)
        #wavdata[k,:read_in_length] = wav_file[0:read_in_length]

        f0, sp, ap = pw.wav2world(sound_data, fs, 5) # 2

        # DEBUG: resynth
        #debug_synth(f0,sp,ap,fs,5)

        #print(sp.shape)
        #print(data_.shape)
        # Because of the 0th order spectra is needed, we use -1 for bin size

        sp = sptk.conversion.sp2mc(sp, bins_1 - 1, alpha)

        # Encode the spectral envelopes
        ap = pw.code_aperiodicity(ap, fs)

        # DEBUG: Decode spectral envelope
        #debug_resynth(f0,sp,ap,fs,5)
        
        # Linear interpolation improved
        puref0 = f0

        # Pure f0set
        read_in_length = np.minimum(puref0.shape[0],max_f0_length)
        puref0set[k,0:read_in_length] = puref0[0:read_in_length]

        # Spectrum
        read_in_length = np.minimum(sp.shape[0],max_f0_length)
        spset[k,0:read_in_length,:] = sp[0:read_in_length,:]


        # Band aperiodicites
        read_in_length = np.minimum(ap.shape[0],max_f0_length)
        apset[k,0:read_in_length,:] = ap[0:read_in_length,:]

        # A buzz is introduced here :(
        #debug_resynth(puref0set[k,:read_in_length],spset[k,:read_in_length,:],apset[k,:read_in_length,:],fs,5)

    if normalisation:

        # Normalise ema feature wise but do not return normaliser object
        for j in range(all_channel):
            scaler_ema = preprocessing.StandardScaler()
            dataset[train_idx,:,j] = scaler_ema.fit_transform(dataset[train_idx,:,j])
            dataset[val_idx,:,j] = scaler_ema.transform(dataset[val_idx,:,j])
            
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

    np.save("processed_comb/dataset_", dataset)
    np.save("processed_comb/puref0set_", puref0set)
    np.save("processed_comb/spset_", spset)
    np.save("processed_comb/apset_", apset)
    np.save("processed_comb/train_idx_", train_idx)
    np.save("processed_comb/val_idx_", val_idx)
    np.save("processed_comb/wavdata", wavdata)

    joblib.dump(scaler_sp, 'processed_comb/scaler_sp_.pkl')
    joblib.dump(scaler_ap, 'processed_comb/scaler_ap_.pkl')

def load_test(delay,percentage=1):
    """Loads the data from the preprocessed numpy arrays

    Keyword arguments:
    - delay - the amount of delay in samples to apply to the output data. The samples at the
beginning are padded with zeroes.
    - percentage - percentage of the dedicated training data to actually use for training. This is useful to change in order to see if model performance is data-limited

    """
    
    dataset = np.load("processed_comb/dataset_.npy")
    #f0set = np.load("processed/f0set_.npy")
    spset = np.load("processed_comb/spset_.npy")
    apset = np.load("processed_comb/apset_.npy")
    puref0set = np.load("processed_comb/puref0set_.npy")
    #scaler_f0 = joblib.load('processed/scaler_f0_.pkl')
    scaler_sp = joblib.load('processed_comb/scaler_sp_.pkl')
    #scaler_ap = joblib.load('processed/scaler_ap_.pkl')
    train_idx = np.load("processed_comb/train_idx_.npy")
    test_idx = np.load("processed_comb/val_idx_.npy")

    # Reduce training id size. It is shuffled by default so it is not reshuffled for brevity
    keep_amount = int(np.ceil(percentage * len(train_idx)))
    train_idx = train_idx[:keep_amount]
    
    ema_test = dataset[test_idx,:,:]
    # Padding f0
    puref0_test = np.pad(puref0set[test_idx,:],((0,0),(delay,0)), mode="constant")[:,:-delay]

    # Padding spectra
    sp_test = np.pad(spset[test_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay,:]

    # Padding ap
    #ap_test = np.pad(apset[test_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay]

    return ema_test, sp_test, None,puref0_test, None, scaler_sp, \
        None
def load_data(delay,percentage=1):
    """Loads the data from the preprocessed numpy arrays

    Keyword arguments:
    - delay - the amount of delay in samples to apply to the output data. The samples at the
beginning are padded with zeroes.
    - percentage - percentage of the dedicated training data to actually use for training. This is useful to change in order to see if model performance is data-limited

    """
    
    dataset = np.load("processed_comb/dataset_.npy")
    spset = np.load("processed_comb/spset_.npy")
    apset = np.load("processed_comb/apset_.npy")
    scaler_sp = joblib.load('processed_comb/scaler_sp_.pkl')
    train_idx = np.load("processed_comb/train_idx_.npy")
    wavdata = np.load("processed_comb/wavdata.npy")
    test_idx = np.load("processed_comb/val_idx_.npy")

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
    
    # Padding spectra
    sp_train = np.pad(spset[train_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay]
    sp_test = np.pad(spset[test_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay,:]
    spset = None
    # Padding ap
    ap_train = np.pad(apset[train_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay]
    ap_test = np.pad(apset[test_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay]
    apset = None
    # Unprocssed f0
    givenf0_train = None
    givenf0_test = None
    givenf0set = None
    return ema_train, ema_test, \
        None, None, \
        sp_train, sp_test, \
        None, None, \
        None, None, \
        wavdata, None, scaler_sp, None

#preprocess_save_combined(normalisation=True,alpha=0.42,max_length=1000,
#                                         fs=16000, val_split=0.1,
#                         noise=False)

