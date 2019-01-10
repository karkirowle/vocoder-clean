import numpy as np
from sklearn.externals import joblib

def delay_signal(signal,delay):
    """
    Wrapper for numpy boilerplate for delaying the signal.
    The idea is that signal can be delay by shifting and zero padding
    in the beginning.
    

    Parameters:
    -----------
    signal: The signal isassumed to be tensors of either
    - 3D (Sample X Time x Channel)
    - 2D (Sample X Time)

    Returns:
    --------
    delayed_signal: the delayed signal same shape as signal
    """

    assert len(signal.shape) < 4, "Invalid signal shape"
    assert len(signal.shape) > 1, "Invalid signal shape"
    assert delay >= 0, "Only positive delay could be applied"

    if (delay == 0):
        return signal
    
    if len(signal.shape) == 2:
        delayed_signal = np.pad(signal,
                                ((0,0),(delay,0)),
                                mode="constant")[:,:-delay]
    else:
        delayed_signal = np.pad(signal,
                                ((0,0),(delay,0),(0,0)),
                                mode="constant")[:,:-delay,:]
    return delayed_signal

def load_test(delay,percentage=1,save_dir="processed_comb2_filtered_2",
              k=0):
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

    train_idx = np.concatenate(train_idx[k])
    test_idx = test_idx[k]
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

import matplotlib.pyplot as plt
def load_compare(delay,save_dir="processed_comb2_filtered_2"):
    delay = 1
    idx = [2132,121,43]

    spset = np.load(save_dir + "/spset_.npy")
    if (delay == 0):
        full_load = spset[idx,:,:]
    else:
        full_load = delay_signal(spset[idx,:,:],delay)
    
    print(full_load.shape)
    single_load = np.zeros((3,1000,82))
    for i in range(3):
        single_temp = np.load(save_dir + "/spset_" + str(idx[i]) + ".npy")
        single_load[i,:,:] = single_temp
    single_load = delay_signal(single_load,1)
    rmse = np.sqrt(np.sum(np.sum((full_load - single_load)**2)))
    print(rmse)
def load_data(delay,percentage=1,save_dir="processed_comb2_filtered_2", k=0):
    
    """
    Loads the dataset for testing AND training.
    
    Parameters:
    -----------
    delay: number of delay to use
    percentage: percentage of the dataset to load.
    save_dir: directory to load the data from
    k: which fold to use form cv folds
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

    train_idx = np.concatenate(train_idx[k])
    test_idx = test_idx[k]
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

import keras
class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """
    
    def __init__(self, options, train, shuffle=True):
        """
        Initialization
        """
        self.in_channel = options["num_features"]
        self.out_channel = options["out_features"]
        self.batch_size = options["batch_size"]
        self.delay = options["delay"]
        self.T = 1000
        self.k = options["k"]
        self.shuffle = shuffle
        self.save_dir = options["save_dir"]
        
        if (train):
            train_idx = np.load(options["save_dir"] + "/train_idx_.npy")
            train_idx = train_idx[self.k]
            self.list_IDs = train_idx
        else:
            test_idx = np.load(options["save_dir"] + "/val_idx_.npy")
            test_idx = test_idx[self.k]
            self.list_IDs = test_idx            

        self.on_epoch_end()
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """
        
        # Initialization
        X = np.empty((self.batch_size, self.T, self.in_channel))
        Y = np.empty((self.batch_size, self.T, self.out_channel))
        

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            Xtemp = np.load(self.save_dir + "/dataset_" + str(ID) + '.npy')
            X[i,:,:] = Xtemp

            # Store class - truncate the end if out_channel is less
            Ytemp = np.load(self.save_dir + "/spset_" + str(ID) + '.npy')
            Y[i,:,:] = Ytemp[:,:self.out_channel]
        Y = delay_signal(Y,self.delay)
        return X, Y

