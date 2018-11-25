
# File for preprocessing the MOCHA-TIMIT dataset
import numpy as np

import matplotlib.pyplot as plt


import soundfile as sf
import sounddevice as sd

import time

from sklearn import preprocessing
from sklearn.externals import joblib
from scipy.signal import decimate
# Speech processing frameworks
import pyworld as pw
import python_speech_features as psf
import pysptk as sptk
from nnmnkwii.preprocessing import interp1d

import tqdm as tqdm

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
    columns[0] = "time"
    columns[1] = "present"

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
            columns[channel_number + 2] = channel_name
            line = clean(f.readline())
        string = f.read()
        data = np.fromstring(string, dtype='float32')
        data_ = np.reshape(data, (-1, len(columns)))

        return data_

# The idea is that the initial preprocessing is run by this function and then the
# generators will yield the delays, so that the delays dont need to be hardsaved

def debug_synth(f0,sp,ap,fs):
    sound = pw.synthesize(f0,sp,ap,fs,2)
    sd.play(sound,fs)
    time.sleep(5)

def debug_resynth(f0_,sp_,ap_,fs,alpha=0.42):
    sp_ = sptk.conversion.mc2sp(sp_, alpha, 1024)
    ap_ = pw.decode_aperiodicity(ap_, fs, 1024)
    sound = pw.synthesize(f0_,sp_,ap_,fs,2)
    sd.play(sound,fs)
    time.sleep(5)
    

def preprocess_save(normalisation=True,alpha=0.42,max_length=2800, fs=16000, val_split=0.2):

    # Conversion to np.array is due to indexing needed later
    files = np.array(glob.glob("dataset/fsew0_*.ema"))
    files2 = np.array(glob.glob("dataset/msak0_*.ema"))
    files = np.concatenate((files,files2))

    total_samples = len(files)

    print("Preprocessing " + str(total_samples) + " samples")
    # Partion the list into training and validation file lists to avoid memory overhead
    indices = np.arange(len(files))
    np.random.shuffle(indices)

    assert val_split < 1 and val_split > 0,"Validation split must be a number on open interval (0,1)"

    validation_size = int(np.ceil(val_split * total_samples))
    val_idx = indices[:validation_size]
    train_idx = indices[validation_size:]

    # Some built-in parameters
    all_channel = 21
    max_audio_length = 89600
    max_f0_length = 2800
    bins_1 = 41
    bins_2 = 1

    # TODO: Corrupted data points treatment
    #sample_list = sample_list - set([40,118,251,268,299,426])
 
    dataset = np.zeros((total_samples,max_length,all_channel))
    puref0set = np.zeros((total_samples,max_f0_length))
    givenf0set = np.zeros((total_samples,max_f0_length))
    f0set = np.zeros((total_samples,max_f0_length))
    spset = np.zeros((total_samples,max_f0_length,bins_1))
    apset = np.zeros((total_samples,max_f0_length,bins_2))
    vset = np.zeros((total_samples,max_f0_length))
    wavdata = np.zeros((total_samples,max_audio_length))
    # Shuffling train_test ids

    for k,fname in enumerate((files)):
        print(k)
        data_ = ema_read(fname)
        
        # We dont need the time and present rows
        read_in_length = np.minimum(data_.shape[0],max_length)
        dataset[k,:read_in_length,:all_channel-1] = data_[:read_in_length,2:]

        # Read wav
        wav_path = fname[:-3 or None] + "wav"
        sound_data, fs = sf.read(wav_path)

        # Read in either to max lenth (truncation) or when data is available (zero padding)
        read_in_length = np.minimum(sound_data.shape[0],max_audio_length)
        wav_file = np.zeros((max_audio_length))
        wav_file[0:read_in_length] = sound_data[0:read_in_length]
        wav_file = wav_file + np.random.normal(0,0.003,wav_file.shape)
        wavdata[k,:read_in_length] = wav_file[0:read_in_length]


        lar_path = fname[:-3 or None] + "lar"
        lar_file, _ = sf.read(lar_path)
        lar_data = np.zeros((max_audio_length))
        lar_data[0:read_in_length] = lar_file[0:read_in_length]
        read_in_length = np.minimum(lar_file.shape[0],max_f0_length)
        dataset[k,:read_in_length,20] = lar_data[0:read_in_length]


        f0, sp, ap = pw.wav2world(wav_file, fs, 2) # 2

        # DEBUG: resynth
        #debug_synth(f0,sp,ap)
        
        # Because of the 0th order spectra is needed, we use -1 for bin size
        sp = sptk.conversion.sp2mc(sp, bins_1 - 1, alpha)

        # Encode the spectral envelopes
        ap = pw.code_aperiodicity(ap, fs)

        # DEBUG: Decode spectral envelope
        #debug_resynth(f0,sp,ap)
        
        # Linear interpolation improved
        puref0 = f0
        f0 = interp1d(np.log(f0),kind="linear")

        # Watch out because f0 is a python list instead of an np aray
        read_in_length = np.minimum(len(f0),max_f0_length)
        f0set[k,0:read_in_length] = f0[0:read_in_length]

        # Pure f0set
        read_in_length = np.minimum(puref0.shape[0],max_f0_length)
        puref0set[k,0:read_in_length] = puref0[0:read_in_length]

        # Spectrum
        read_in_length = np.minimum(sp.shape[0],max_f0_length)
        spset[k,0:read_in_length,:] = sp[0:read_in_length,:]

        # Band aperiodicites
        read_in_length = np.minimum(ap.shape[0],max_f0_length)
        apset[k,0:read_in_length,:] = ap[0:read_in_length,:]

        f0, _, _ = pw.wav2world(lar_data, fs, 2)
        read_in_length = np.minimum(spset.shape[1],max_f0_length)
        givenf0set[k,0:read_in_length] = f0[0:read_in_length]

    if normalisation:
        scaler_f0 = preprocessing.StandardScaler()
        f0set[train_idx,:] = scaler_f0.fit_transform(f0set[train_idx,:])
        f0set[val_idx,:] = scaler_f0.transform(f0set[val_idx,:])

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

    np.save("dataset_", dataset)
    np.save("puref0set_", puref0set)
    np.save("f0set_", f0set)
    np.save("spset_", spset)
    np.save("apset_", apset)
    np.save("vset_", vset)
    np.save("givenf0set_", givenf0set)
    np.save("train_idx_", train_idx)
    np.save("val_idx_", val_idx)
    np.save("wavdata", wavdata)

    joblib.dump(scaler_f0, 'scaler_f0_.pkl')
    joblib.dump(scaler_sp, 'scaler_sp_.pkl')
    joblib.dump(scaler_ap, 'scaler_ap_.pkl')

def load_test(delay,percentage=1):
    """Loads the data from the preprocessed numpy arrays

    Keyword arguments:
    - delay - the amount of delay in samples to apply to the output data. The samples at the
beginning are padded with zeroes.
    - percentage - percentage of the dedicated training data to actually use for training. This is useful to change in order to see if model performance is data-limited

    """
    
    dataset = np.load("dataset_.npy")
    f0set = np.load("f0set_.npy")
    spset = np.load("spset_.npy")
    apset = np.load("apset_.npy")
    givenf0set = np.load("givenf0set_.npy")
    scaler_f0 = joblib.load('scaler_f0_.pkl')
    scaler_sp = joblib.load('scaler_sp_.pkl')
    scaler_ap = joblib.load('scaler_ap_.pkl')
    train_idx = np.load("train_idx_.npy")
    test_idx = np.load("val_idx_.npy")

    # Reduce training id size. It is shuffled by default so it is not reshuffled for brevity
    keep_amount = int(np.ceil(percentage * len(train_idx)))
    train_idx = train_idx[:keep_amount]
    
    ema_test = dataset[test_idx,:,:]
    # Padding f0
    f0_test = np.pad(f0set[test_idx,:],((0,0),(delay,0)), mode="constant")[:,:-delay]

    # Padding spectra
    sp_test = np.pad(spset[test_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay,:]

    # Padding ap
    ap_test = np.pad(apset[test_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay]

    # Unprocssed f0
    givenf0_test = np.pad(givenf0set[test_idx,:],((0,0),(delay,0)), mode="constant")[:,:-delay]
    

    return ema_test, sp_test, ap_test,givenf0_test, scaler_f0, scaler_sp, \
        scaler_ap
def load_data(delay,percentage=1):
    """Loads the data from the preprocessed numpy arrays

    Keyword arguments:
    - delay - the amount of delay in samples to apply to the output data. The samples at the
beginning are padded with zeroes.
    - percentage - percentage of the dedicated training data to actually use for training. This is useful to change in order to see if model performance is data-limited

    """
    
    dataset = np.load("dataset_.npy")
    vset = np.load("vset_.npy")
    f0set = np.load("f0set_.npy")
    spset = np.load("spset_.npy")
    apset = np.load("apset_.npy")
    givenf0set = np.load("givenf0set_.npy")
    scaler_f0 = joblib.load('scaler_f0_.pkl')
    scaler_sp = joblib.load('scaler_sp_.pkl')
    scaler_ap = joblib.load('scaler_ap_.pkl')
    train_idx = np.load("train_idx_.npy")
    wavdata = np.load("wavdata.npy")
    test_idx = np.load("val_idx_.npy")

    # Reduce training id size. It is shuffled by default so it is not reshuffled for brevity
    keep_amount = int(np.ceil(percentage * len(train_idx)))
    train_idx = train_idx[:keep_amount]
    
    # EMA and LAR partition
    ema_train = dataset[train_idx,:,:]
    ema_test = dataset[test_idx,:,:]
    # Padding f0
    f0_train = np.pad(f0set[train_idx,:],((0,0),(delay,0)), mode="constant")[:,:-delay]
    f0_test = np.pad(f0set[test_idx,:],((0,0),(delay,0)), mode="constant")[:,:-delay]

    # Padding spectra
    sp_train = np.pad(spset[train_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay]
    sp_test = np.pad(spset[test_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay,:]

    # Padding ap
    ap_train = np.pad(apset[train_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay]
    ap_test = np.pad(apset[test_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay]

    # Unprocssed f0
    givenf0_train = np.pad(givenf0set[train_idx,:],((0,0),(delay,0)), mode="constant")[:,:-delay]
    givenf0_test = np.pad(givenf0set[test_idx,:],((0,0),(delay,0)), mode="constant")[:,:-delay]

    return ema_train, ema_test, \
        f0_train, f0_test, \
        sp_train, sp_test, \
        ap_train, ap_test, \
        givenf0_train, givenf0_test, \
        wavdata, scaler_f0, scaler_sp, scaler_ap
