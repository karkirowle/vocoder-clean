import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import sounddevice as sd

import time

from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.externals import joblib
from scipy.signal import decimate,savgol_filter,resample

# Speech processing frameworks
import pyworld as pw
import python_speech_features as psf
import pysptk as sptk
from nnmnkwii.preprocessing import interp1d, modspec_smoothing, delta_features
from nnmnkwii.metrics import melcd
from nnmnkwii.paramgen import mlpg

# Own stuff
import data_loader
import audio as audio

import tqdm as tqdm
import pandas as pd
from scipy.io import wavfile

import glob

import os        
import sys


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
    Aldo does the preprocessing based on the MNGU0 dataset needs

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
    # TODO: remove sav-gol
  #  for k in range(data.shape[1]):
   #     data[:,k] = savgol_filter(data[:,k], 51, 3)
    return data

def ema_read_tor(fname,start_calib):
    """
    Reads in tor dataset
    """

    pos_number = 3+2+2 
    channel_number = 12
    columns = {}
    columns["time"] = 0
    columns["present"] = 1

    #raw = np.fromfile(fname)
    raw = np.fromfile(fname, dtype=np.dtype('<f'))
    data = np.reshape(raw, (-1, pos_number*channel_number))
    
    # Channel list
    # ---------------
    # 1 - Tongue back
    # 2 - Tongue middle
    # 3 - Tongue tip
    # 4 - Forehead
    # 5 - Bridge of the nose
    # 6 - Upper lip
    # 7 - Lower lip
    # 8 - Lower incisor
    # 9 - Left lip
    # 10 - Right lip
    # 11 - Left ear
    # 12 - Right ear

    pos = [0,2]
    channel_idx = [0,1,2,7,4,5,6]
    pos_idx = []
    for channel_id in channel_idx:
        for k in pos:
            pos_idx.append(channel_id * pos_number + k)

    temp = data[:,pos_idx]

    # Inverting axes
    temp[:,::2] = -temp[:,::2]

    if start_calib == []:
        start_calib = temp[0,:]
    else:
        diff = temp[0,:] - start_calib
        temp = temp - diff

    #plt.plot(temp)
    #plt.show()
    for k in range(temp.shape[1]):
       temp[:,k] = savgol_filter(temp[:,k], 51, 3)
    #t = 0.3
    #temp_diff = np.diff(temp,axis=0)
    #temp_diff[temp_diff > t] = 0
    #temp_diff[temp_diff < -t] = 0

    #temp = np.cumsum(np.concatenate((temp[[0],:],temp_diff),axis=0),axis=0)
        
    return temp, start_calib

import matplotlib.animation as animation
def visualise_torgo():
    """
    Quick visualisation of the TORGO dataet
    """
    
    files2_t1 = np.array(sorted(glob.glob("dataset6/*.pos")))
    print(len(files2_t1))
    samples = len(files2_t1)
    channels = 7

    data = np.zeros((samples,1000,channels*2))
    start_calib = []

    for idx,fname in enumerate(files2_t1):
        temp, start_calib = ema_read_tor(fname,start_calib)
        read_in_length = np.minimum(temp.shape[0],1000)
        data[idx,:read_in_length,:] = temp[:read_in_length,:]
        #plt.plot(data[idx,:])
        #plt.show()
    numFrames = data.shape[1]
    fig = plt.figure()

    scats = []
    for channel_idx in range(channels):
        scats.append(plt.scatter(data[:,0,channel_idx * 2],
                                data[:,0,channel_idx * 2 + 1]))
    ani = animation.FuncAnimation(fig, update_plot, frames=range(numFrames),
                                  fargs=(data, scats))
    plt.legend(['tongue back', 'tongue middle', 'tongue tip',
                'lower incisor', 'nose',
                'uppper lip', 'lower lip'])
    plt.xlim([-50,50])
    plt.ylim([-50,50])
    plt.show()

def update_plot(i,data,scats):

    for idx,scat in enumerate(scats):
        scat.set_offsets(data[:,i,[idx * 2, idx * 2 + 1]])
    return scats,

def train_val_split(files,val_split, k=10):
    """
    Partitions the list into training and validation

    Parameters:
    -----------
    files: concatenated list with filenames
    val_split: percentage of validation split
    k: number of folds in CV to make, must be smaller than len(files)
    Returns:
    ---------
    train_idx: K-elements list of numpy array
    val_idx: K-element lis of numpy array
    """
    total_samples = len(files)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)


    assert val_split < 1 and val_split > 0, \
        "Validation split must be a number on open interval (0,1)"
    assert k < total_samples, \
        "It is not possible to make more folds than samples"

    split_indices = k_split(indices,k)
    train_idx = []
    val_idx = []

    for i in range(k):
        idx_in = [j for j in range(k) if j != i]

        train_id = []
        
        for l in idx_in:
            train_id.append(split_indices[l])

        
        val_id = split_indices[i]

        train_id = np.array(np.concatenate(train_id))
        val_id = np.array(val_id)

        train_idx.append(train_id)
        val_idx.append(val_id)


    return train_idx,val_idx


def k_split(seq, k):
    """
    Partitions a numpy array into k parts
    
    Because the parts might be not equal a list of np arrays are returned
    
    Parameters:
    -----------
    seq - sequence of numbers
    k - number of parts

    Returns:
    --------
    out - list of numpy arrays
    """
    

    assert len(seq) > k, \
        "Seqence has to have more elements than shatters"
    assert k > 0

    avg = len(seq) / float(k)
    out = []
    last = 0.0

    i = 1
    while last < len(seq):
        if (i == k):
            out.append(seq[int(last):])
            last = len(seq)
        else:
            out.append(seq[int(last):int(last + avg)])
        last += avg
        i += 1
    

    return out

def nan_check(dataset):
    """
    Checks a numpy array for nans and prints a "warning"

    Parameters:
    -----------
    dataset: numpy array
    """
    if np.isnan(dataset).any():
        dataset[np.isnan(dataset)] = 0
        print("Warning! Zeroed a NaN")

def figure_reproduction():
    """
    Generates the figures for the paper

    """

    files = np.array(sorted(glob.glob("dataset2/*.ema")))
    total_samples = len(files)
    electrodes = 14
    init_pos = np.zeros((total_samples,electrodes))
    labels = ["T3", "T2", "T1", "Jaw", "Upper lip", "Lower lip"]
    for i,fname in enumerate(files):
        data = ema_read(fname)
        init_pos[i,:] = data[0,:]

    sns.set_style("white")
    idx = [0,1,2,3,5,6]
    markers = [".", "v", "^", "<","<", "s","+"]
    for label in idx:
        plt.scatter(init_pos[:,2*label],
                    init_pos[:,2*label+1],
                    marker=markers[label],
                    color="black")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.legend(labels)
    plt.title("Initial position of electrodes in MNGU0 dataset")
    plt.figure(num=1, figsize=(4.2,4.2), dpi=80, facecolor="w", edgecolor="k")
    #plt.show()
    plt.savefig("paper/init_pos.pgf")

    return 0
def sp_delta_generation(sp,mfcc_bins,alpha,hop_length):
    """
    Takes a sound file and generates the MFCC static and delta 
    features.

    Parameters:
    -----------
    sp: numpy array containing spectrum for each analysis window
    mfcc_bins: number of MFCC bins
    alpha: pre-emphasis filter coefficients
    hop_length: ?

    Returns:
    --------
    sp_delta: MFCC and delta features in one combined numpy array
    """
    # Because of the 0th order spectra is needed, we use -1 for bin
    sp = sptk.conversion.sp2mc(sp, mfcc_bins - 1, alpha)

    # TODO: what the hell is the hop length
    s_sp = modspec_smoothing(sp,hop_length)

    # Getting the delta features
    windows = [
        (0, 0, np.array([1.0])),            # static
        (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
    ]
    sp_delta = delta_features(s_sp,windows)

    return sp_delta

def mlpg_postprocessing(mfcc, bins_1, scaler_sp):
    """
    Takes the static and delta features, converts back and unnormalises

    Parameters:
    ----------
    mfcc: (N x T X bins_1) array
    bins_1: number of static + delta coefficients + power 
    scaler_sp: normalisation object

    Return:
    -------
    MFCC features after maximum likelihood parameter generations
    """

    windows = [
        (0, 0, np.array([1.0])),            # static
        (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
    ]

    N = mfcc.shape[0]
    T = mfcc.shape[1]

    mlpg_generated = np.zeros((N,T,bins_1))
    
    for i in range(N):
        mlpg_generated[i,:,:] = mlpg(mfcc[i,:,:],
                                     np.ones((T,bins_1 * 2)),
                                     windows)
    for i in range(len(scaler_sp)):
        mlpg_generated[:,:,i] = scaler_sp[i].inverse_transform(mlpg_generated[:,:,i])
    
    return mlpg_generated

def fetch_validation_size(options,label,model):
    """
    Parameters:
    -----------
    options - every option needed for the data loader
    label - which dataset 
    model - model we trained our dataset with 
    Returns:
    --------
    size of the validation set
    """
    # Unshuffled, unswapped validation set
    val_gen = data_loader.DataGenerator(options,
                                        False,
                                        False,
                                        False,
                                        False,
                                        label=label)

    # Run predictions for all minibatch
    # which fill infer the full batch size 
    sp_test_hat = model.predict_generator(val_gen)
    validation_size = sp_test_hat.shape[0]
    return validation_size

def evaluate_validation(model,options,sbin,label):
    """
    Parameters:
    --------------
    model - keras model to use for evaluation
    options["save_dir"] - where to get the normaliser object from
    options["k"] - which fold to use
    options["batch_size"] - size of the validation set
    label - which dataset
    sbin - cepstral bin size

    Return
    --------
    MCD - mel cepstral distortion


    """

    # Full validation set is compared, so first we infer size

    N = fetch_validation_size(options,label,model)
    options["batch_size"] = N
    val_gen = data_loader.DataGenerator(options,False,False,False,False,
                                        label=label)
    sp_train, sp_test, _ = val_gen.__getitem__(0)
    sp_test_hat = model.predict_generator(val_gen)

    scaler_sp = joblib.load(options["save_dir"] + '/scaler_sp_.pkl')

    # Perform MLPG
    mlpg_generated = mlpg_postprocessing(sp_test_hat,
                                          sbin,
                                          scaler_sp)

    # Perform destandardisation
    sp_test_u = np.copy(sp_test_hat)
    sp_test_hat_u = np.copy(sp_test_hat)
    for i in range(len(scaler_sp)):
        sp_test_u[:,:,i] = scaler_sp[i].inverse_transform(sp_test[:,:,i])
        sp_test_hat_u[:,:,i] = scaler_sp[i].inverse_transform(sp_test_hat[:,:,i])

    mcd = melcd(mlpg_generated,sp_test_u[:,:,:sbin])
    return mcd

def f0_process(f0,linear=True):
    """
    
    Parameters:
    -----------
    f0 signal
    linear - Linear interpolation for values where np.log(0) = -np.inf
    If every value is 0, it returns 0 for all

    Return:
    -----------
    the log interpolated f0
    """

    if (linear):
        f0_log = np.log(f0,
                        out=-np.inf*np.ones_like(f0),
                        where=(f0!=0))

        if (np.isinf(f0_log).all()):
            f0 = np.zeros_like(f0)
        else:
            f0 = interp1d(f0_log,kind="slinear")
    else:

        # Calculate running average for all points

        # Exponential decay function
        f0_ = pd.Series(np.log(f0,
                               out=-np.nan*np.ones_like(f0),
                               where=(f0!=0)))
        f0_ = f0_.interpolate(limit_area="inside",method="polynomial", order=2)
        plt.plot(f0_)
        plt.show()
    return f0

def check_paired():
    paths = ["dataset2/*.ema", "dataset/msak*.ema",
             "dataset/fsew*.ema", "dataset3/*.pos",
             "dataset4/*.pos", "dataset5/*.pos",
             "dataset6/*.pos", "dataset7/*.pos",
             "dataset8/*.pos", "dataset9/*.pos",
             "dataset10/*.pos", "dataset11/*.pos",
             "dataset12/*.pos"]

    files = np.array([np.array(sorted(glob.glob(path))) for path in paths])
    files = np.concatenate(files)

    for fname in files:
        fname_wo_extension = fname[:-3 or None]
        wav_path = fname_wo_extension + "wav"
        sound_data, fs = sf.read(wav_path)
    print("Everything is paired! Nice work")
    
def preprocess_save_combined(alpha=0.42,
                             max_length=1000, fs=16000, val_split=0.2,
                             noise=False,combined=True, bins_1 = 41,
                             bins_2 = 1, normalisation_input = True,
                             normalisation_output = True,
                             channel_number = 14,
                             factor = 80,
                             save_dir = "processed_comb_test_4_padded"):
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

    check_paired()
    # Fetching and shuffling the appropriate file lists
    if combined:
        paths = ["dataset2/*.ema", "dataset/msak*.ema",
                 "dataset/fsew*.ema", "dataset3/*.pos",
                 "dataset4/*.pos", "dataset5/*.pos",
                 "dataset6/*.pos", "dataset7/*.pos",
                 "dataset8/*.pos", "dataset9/*.pos",
                 "dataset10/*.pos", "dataset11/*pos",
                 "dataset12/*.pos", "dataset13/*pos"]

        files = np.array([np.array(sorted(glob.glob(path))) for path in paths])
        files = np.concatenate(files)
    else:
        files = np.array(glob.glob("dataset2/*.ema"))

    np.random.shuffle(files)
    total_samples = len(files)
    print("Preprocessing " + str(total_samples) + " samples")

    all_idx = range(total_samples)

    # Last hundred is for testing
    files2 = files[:-100]
    print(files2.shape)
    train_idx, val_idx = train_val_split(files2,0.2)

    # Preallocation of memory
    dataset = np.zeros((total_samples,max_length,channel_number+1))
    puref0set = np.zeros((total_samples,max_length))
    spset = np.zeros((total_samples,max_length,bins_1 * 2))
    apset = np.zeros((total_samples,max_length,bins_2))

    # Which ID correspond to which category/dataset, male, female
    cat_id = { "male": [],
               "female": [],
               "mngu0": [],
               "d3": [],
               "d4": [],
               "d5": [],
               "d6": [],
               "d7": [],
               "d8": [],
               "d9": [],
               "d10": [],
               "d11": [],
               "d12": [],
               "d13": []}

    
    start_calib = []
    # Append the appropriate id and read files while showing progress
    for k,fname in tqdm.tqdm(enumerate((files)), total=len(files)):

        if "mngu0" in fname:
            data = ema_read(fname)
            cat_id["mngu0"].append(k)
        else:
            if "fsew" in fname:
                cat_id["female"].append(k)
                data = ema_read_mocha(fname)
                # Resample so the datasets have same sampling frequency
                data = resample(data,int(np.ceil(data.shape[0]*2/5)))
            if "msak" in fname:
                cat_id["male"].append(k)
                data = ema_read_mocha(fname)
                # Resample so the datasets have same sampling frequency
                data = resample(data,int(np.ceil(data.shape[0]*2/5)))
            shorts = ["d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
                      "d11", "d12","d13"]
            for short in shorts:
                if short in fname:
                    cat_id[short].append(k)
                    data, start_calib = ema_read_tor(fname,start_calib)
                    
        # We don't need the time and present rows
        read_in_length = np.minimum(data.shape[0],max_length)
        dataset[k,:read_in_length,:-1] = data[:read_in_length,:]

        # Repeating last elements if the samples are not the same
        if (max_length > data.shape[0]):
            dataset[k,data.shape[0]:,:-1] = data[data.shape[0]-1,:]

        fname_wo_extension = fname[:-3 or None]
        wav_path = fname_wo_extension + "wav"
        sound_data, fs = sf.read(wav_path)
        if (len(sound_data.shape) == 2):
            sound_data = sound_data[:,1].copy(order='C')

        f0, sp, ap = pw.wav2world(sound_data, fs, 5) # 2

        
        # The general way is to either truncate or zero-pad
        T = f0.shape[0]
        read_in_length = np.minimum(T,max_length)
        dataset[k,:read_in_length,channel_number] = f0_process(f0[:read_in_length])
        dataset[k,:,channel_number] = f0_process(dataset[k,:,channel_number])
        sp_delta = sp_delta_generation(sp,bins_1,alpha,200)

        ap = pw.code_aperiodicity(ap, fs)

        puref0set[k,:read_in_length] = f0[:read_in_length]
        spset[k,:read_in_length,:] = sp_delta[:read_in_length,:]

        # Also doing the repetition for the spectra
        if (max_length > T):
            spset[k,T:,:] = np.mean(sp_delta[T-5:T-1,:],axis=0)

        apset[k,:read_in_length,:] = ap[:read_in_length,:]

    if normalisation_input:

        # Normalise the articulographs differently for different references
        cats = ["male", "female", "mngu0","d3","d4","d5","d6","d7",
                "d8","d9","d10","d11","d12","d13"]
        
        # Normalise ema feature wise but do not return normaliser object
        for j in range(channel_number + 1):
            for cat in cats:
                scaler_ema = preprocessing.StandardScaler()
                tidx = cat_id[cat]
                dataset[tidx,:,j] = scaler_ema.fit_transform(dataset[tidx,:,j])
                
    if normalisation_output:
        # Spectrum scalers
        scaler_sp = []
        for k in range(bins_1):
            scaler_sp.append(preprocessing.StandardScaler())
            spset[all_idx,:,k] = scaler_sp[k].fit_transform(spset[all_idx,:,k])

        # Aperiodicities scalers
        scaler_ap = []
        for k in range(bins_2):
            scaler_ap.append(preprocessing.StandardScaler())
            apset[all_idx,:,k] = scaler_ap[k].fit_transform(apset[all_idx,:,k])

    # Create dir
    if not os.path.isdir(save_dir):
            os.mkdir(save_dir,0o755)
    # Savin on a per-indice base
    for k,fname in tqdm.tqdm(enumerate((files)), total=len(files)):
        np.save(save_dir + "/dataset_" + str(k), dataset[k,:,:])
        np.save(save_dir + "/puref0set_" + str(k), puref0set[k,:])
        np.save(save_dir + "/spset_" + str(k), spset[k,:,:])
        np.save(save_dir + "/apset_" + str(k), apset[k,:,:])

    np.save(save_dir + "/dataset", dataset)
    np.save(save_dir + "/train_idx_", train_idx)
    np.save(save_dir + "/val_idx_", val_idx)
    np.save(save_dir + "/cat_id", cat_id)
    joblib.dump(scaler_sp, save_dir + '/scaler_sp_.pkl')
    joblib.dump(scaler_ap, save_dir + '/scaler_ap_.pkl')

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", action="store_true")
    parser.add_argument("--fig", action="store_true")
    parser.add_argument("--vis_tor", action="store_true")


    args = parser.parse_args()

    if args.proc:
        preprocess_save_combined()
    if args.fig:
       figure_reproduction()
    if args.vis_tor:
        visualise_torgo()
