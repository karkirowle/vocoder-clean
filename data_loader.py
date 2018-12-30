import numpy as np
from sklearn.externals import joblib

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
    assert len(signal.shape) > 1, "Invalid signal shape"
    if len(signal.shape) == 2:
        delayed_signal = np.pad(signal,
                                ((0,0),(delay,0)),
                                mode="constant")[:,:-delay]
    else:
        delayed_signal = np.pad(signal,
                                ((0,0),(delay,0),(0,0)),
                                mode="constant")[:,:-delay,:]
    return delayed_signal

def load_test(delay,percentage=1,save_dir="processed_comb2_filtered_2"):
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

    sp_test_u = np.copy(sp_test)
    bap_gt_u = np.copy(ap_test)

    for i in range(len(scaler_sp)):
        sp_test_u[:,:,i] = scaler_sp[i].inverse_transform(sp_test[:,:,i])

    for i in range(len(scaler_ap)):
        bap_gt_u[:,:,i] = scaler_ap[i].inverse_transform(ap_test[:,:,i])

    return ema_test, sp_test, ap_test,puref0_test, scaler_sp, \
        scaler_ap, sp_test_u, bap_gt_u

def load_data(delay,percentage=1,save_dir="processed_comb2_filtered_2"):
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
    
    sp_train = delay_signal(spset[train_idx,:,:],delay)
    sp_test = delay_signal(spset[test_idx,:,:],delay)
    spset = None

    ap_train = delay_signal(apset[train_idx,:,:],delay)
    ap_test = delay_signal(apset[test_idx,:,:],delay)
    apset = None

    return ema_train, ema_test, \
        sp_train, sp_test, scaler_sp
