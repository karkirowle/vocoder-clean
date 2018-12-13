
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
    files_m = np.array(sorted(glob.glob("dataset/msak*.ema")))
    files_f = np.array(sorted(glob.glob("dataset/fsew*.ema")))
    files2 = np.array(sorted(glob.glob("dataset2/*.ema")))
    points1_m = np.zeros((len(files_m),14))
    points1_f = np.zeros((len(files_f),14))
    points2 = np.zeros((len(files2),14))

    for idx,fname in enumerate(files_m):
        data_ = ema_read_mocha(fname)
        points1_m[idx,:] = data_[0,:]

    for idx,fname in enumerate(files_f):
        data_ = ema_read_mocha(fname)
        points1_f[idx,:] = data_[0,:]
    
    for idx,fname in enumerate(files2):
        data__ = ema_read(fname)
        points2[idx,:] = data__[0,:]

    #sns.set(style="darkgrid")
    #plotlist = [0,1,2,3,4,5,6,7,10,11,12,13]
    #plt.plot(data__[:,plotlist])
    #plt.xlabel("time [s]")
    #plt.ylabel("electrode positions")
    #plt.title("Example articulatory recording")
    #plt.legend(['T3 X','T3 Y',
    #            'T2 X', 'T2 Y',
    #            'T1 X', 'T1 Y',
    #            'Lower incisor X', 'Lower incisor Y',
    #            'Upper lip X', 'Upper lip Y',
    #            'Lower lip X', 'Lower lip Y'])
    #plt.show()

    sns.set(style="darkgrid")
    plt.subplot(1,2,1)
    plotlist = [0,1,2,3,5,6]
    for i in plotlist:
        #ax = sns.kdeplot(points1[:,2*i], points1[:,2*i+1],
        #                 shade=True, shade_lowest=False)
        #plt.scatter(points1_m[:,2*i],points1_m[:,2*i+1],marker="+")
        plt.scatter(points1_f[:,2*i],points1_f[:,2*i+1],marker="+")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Initial position of electrodes in MOCHA-TIMIT dataset")
    plt.legend(['T3','T2','T1','Lower incisor','Upper lip', 'Lower lip'])

    plt.subplot(1,2,2)
    sep = 300
    for i in plotlist:
        plt.scatter(points2[:sep,2*i],points2[:sep,2*i+1],marker="+")
        plt.scatter(points2[sep:,2*i],points2[sep:,2*i+1],marker="+")
        plt.xlabel("x position")
        plt.ylabel("y position")
    plt.title("Initial position of electrodes in MNGU0 dataset")
    plt.legend(['T3','T2','T1','Lower incisor','Upper lip', 'Lower lip'])
    plt.show()

    points1_m_mean = np.mean(points1_m,axis=0)
    points1_f_mean = np.mean(points1_f,axis=0)
    points2_mean = np.mean(points2,axis=0)
    scale_m = np.std((points2 - points2_mean),axis=0)/np.std((points1_m - points1_m_mean),axis=0)
    scale_f = np.std((points2 - points2_mean),axis=0)/np.std((points1_f - points1_f_mean),axis=0)
    
    points1_m = (points1_m - points1_m_mean)*scale_m + points2_mean
    points1_f = (points1_f - points1_f_mean)*scale_f + points2_mean

    for i in plotlist:
        plt.scatter(points1_m[:,2*i],points1_m[:,2*i+1],marker="+")
        plt.scatter(points1_f[:,2*i],points1_f[:,2*i+1],marker="+")
        plt.scatter(points2[:,2*i],points2[:,2*i+1],marker="+")

    plt.show()
    return scale_m,scale_f,points1_m_mean,points1_f_mean,points2_mean
    
def debug_synth(f0,sp,ap,fs,an=2):
    sound = pw.synthesize(f0,sp,ap,fs, an)
    sd.play(sound,fs)
    sd.wait()

def debug_resynth(f0_,sp_,ap_,fs,an=2,alpha=0.42,fftbin=1024):
    sp_ = sptk.conversion.mc2sp(sp_, alpha, fftbin)
    ap_ = pw.decode_aperiodicity(ap_, fs, fftbin)
    sound = pw.synthesize(f0_,sp_,ap_,fs,an)
    sd.play(sound*3,fs)
    sd.wait()

def save_resynth(fname,f0_,sp_,ap_,fs,an=2,alpha=0.42,fftbin=1024):
    sp_ = sptk.conversion.mc2sp(sp_, alpha, fftbin)
    ap_ = pw.decode_aperiodicity(ap_, fs, fftbin)
    sound = pw.synthesize(f0_,sp_,ap_,fs,an)
    wavfile.write(fname,fs,sound*3)

#save_dir = "processed_mngu0_filtered"
save_dir = "processed_comb2_filtered"

def preprocess_save_combined(normalisation=True,alpha=0.42,
                             max_length=1000, fs=16000, val_split=0.2,
                             noise=False,combined=False, bins_1 = 41,
                             bins_2 = 1, normalisation_input = True,
                             normalisation_output = True):
    """
    Normalisation
    """

    if combined:
        files1 = np.array(sorted(glob.glob("dataset2/*.ema")))
        files2_m = np.array(sorted(glob.glob("dataset/msak*.ema")))
        files2_f = np.array(sorted(glob.glob("dataset/fsew*.ema")))
        files = np.concatenate((files1,files2_m,files2_f))
    else:
        files = np.array(glob.glob("dataset2/*.ema"))
    np.random.shuffle(files)
    #files = files[:100]
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
    
    dataset = np.zeros((total_samples,max_length,all_channel+1))
    puref0set = np.zeros((total_samples,max_f0_length))
    spset = np.zeros((total_samples,max_f0_length,bins_1 * 2))
    apset = np.zeros((total_samples,max_f0_length,bins_2))
    scale_m,scale_f,points1_m_mean,points1_f_mean,points2_mean = data_combine()

    male_id = []
    female_id = []
    mngu0_id = []
    
    # Shuffling train_test ids
    for k,fname in enumerate((files)):
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
            data_ = resample(data_,int(np.ceil(data_.shape[0]*2/5)))

        #plt.plot(data_)
        #plt.show()
        # We dont need the time and present rows
        read_in_length = np.minimum(data_.shape[0],max_length)
        dataset[k,:read_in_length,:-1] = data_[:read_in_length,:]

        # Repeating last elements
        if (max_length > data_.shape[0]):
            dataset[k,data_.shape[0]:,:-1] = data_[data_.shape[0]-1,:]
        if np.isnan(dataset).any():
            dataset[np.isnan(dataset)] = 0
            print("Warning! Zeroed a NaN")

        # Read wav
        wav_path = fname[:-3 or None] + "wav"
        sound_data, fs = sf.read(wav_path)

        # DEBUG: SD play check
        #sd.play(sound_data,fs)
        #sd.wait()
        # Read in either to max lenth (truncation) or when data is available (zero padding)

        f0, sp, ap = pw.wav2world(sound_data, fs, 5) # 2

        # SP Trajectory smoothing

        read_in_length = np.minimum(f0.shape[0],max_f0_length)
        dataset[k,:read_in_length,all_channel] = f0[:read_in_length]
        # DEBUG: resynth
        #debug_synth(f0,sp,ap,fs,5)

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
        #debug_resynth(f0,sp_dec,ap,fs,5)

        # Pure f0set
        read_in_length = np.minimum(f0.shape[0],max_f0_length)
        puref0set[k,0:read_in_length] = f0[0:read_in_length]

        # Spectrum
        read_in_length = np.minimum(sp.shape[0],max_f0_length)
        spset[k,0:read_in_length,:] = sp_delta[0:read_in_length,:]

        # Band aperiodicites
        read_in_length = np.minimum(ap.shape[0],max_f0_length)
        apset[k,0:read_in_length,:] = ap[0:read_in_length,:]

        #sp_dec = mlpg(spset[k,:read_in_length,:],dummy_var[:read_in_length,:],windows)
        #debug_resynth(puref0set[k,:read_in_length],sp_dec,apset[k,:read_in_length,:],fs,5)

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
        for j in range(7):
            plt.scatter(dataset[train_male,0,2*j],dataset[train_male,0,2*j + 1])
            plt.scatter(dataset[train_female,0,2*j],dataset[train_female,0,2*j + 1])
            plt.scatter(dataset[train_mngu0,0,2*j],dataset[train_mngu0,0,2*j + 1])
            plt.show()

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

def load_test(delay,percentage=1):
    """Loads the data from the preprocessed numpy arrays

    Keyword arguments:
    - delay - the amount of delay in samples to apply to the output data. The samples at the
beginning are padded with zeroes.
    - percentage - percentage of the dedicated training data to actually use for training. This is useful to change in order to see if model performance is data-limited

    """
    
    dataset = np.load(save_dir + "/dataset_.npy")
    #f0set = np.load("processed/f0set_.npy")
    spset = np.load(save_dir + "/spset_.npy")
    apset = np.load(save_dir + "/apset_.npy")
    puref0set = np.load(save_dir + "/puref0set_.npy")
    #scaler_f0 = joblib.load('processed/scaler_f0_.pkl')
    scaler_sp = joblib.load(save_dir + '/scaler_sp_.pkl')
    scaler_ap = joblib.load(save_dir + '/scaler_ap_.pkl')
    train_idx = np.load(save_dir + "/train_idx_.npy")
    test_idx = np.load(save_dir + "/val_idx_.npy")

    # Reduce training id size. It is shuffled by default so it is not reshuffled for brevity
    keep_amount = int(np.ceil(percentage * len(train_idx)))
    train_idx = train_idx[:keep_amount]
    
    ema_test = dataset[test_idx,:,:]
    # Padding f0
    puref0_test = np.pad(puref0set[test_idx,:],((0,0),(delay,0)), mode="constant")[:,:-delay]

    # Padding spectra
    sp_test = np.pad(spset[test_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay,:]

    # Padding ap
    ap_test = np.pad(apset[test_idx,:,:],((0,0),(delay,0),(0,0)), mode="constant")[:,:-delay]

    return ema_test, sp_test, ap_test,puref0_test, None, scaler_sp, \
        scaler_ap
def load_data(delay,percentage=1):
    """Loads the data from the preprocessed numpy arrays

    Keyword arguments:
    - delay - the amount of delay in samples to apply to the output data. The samples at the
beginning are padded with zeroes.
    - percentage - percentage of the dedicated training data to actually use for training. This is useful to change in order to see if model performance is data-limited

    """
    
    dataset = np.load(save_dir + "/dataset_.npy")
    spset = np.load(save_dir + "/spset_.npy")
    apset = np.load(save_dir + "/apset_.npy")
    scaler_sp = joblib.load(save_dir + '/scaler_sp_.pkl')
    train_idx = np.load(save_dir + "/train_idx_.npy")
#    wavdata = np.load(save_dir + "/wavdata.npy")
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
        None, None, scaler_sp, None

#preprocess_save_combined(normalisation=True,alpha=0.42,max_length=1000,
#                                                                  fs=16000, val_split=0.1,
#                         noise=False,combined=True)

#data_combine()
