import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import keras

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

def load_puref0(save_dir,k):
    """
    Loads just the unprocessed f0 values for testing
    """
    
    test_idx = np.load(save_dir + "/val_idx_.npy")
    test_idx = test_idx[k]

    # Infer the dataset size run-time
    example = np.load(save_dir + "/puref0set_1.npy")
    puref0set = np.zeros((len(test_idx),example.shape[0]))

    for i, idx in enumerate(test_idx):
        puref0set[i,:] = np.load(save_dir + "/puref0set_" +
                                   str(idx) +
                                   ".npy")

    return puref0set

def load_scalersp(save_dir):
    """
    Loads scaler object

    Parameters:
    -----------
    save_dir - directory to load data from
    """
    return joblib.load(save_dir + "/scaler_sp_.pkl")

def load_bap(save_dir,k):
    """
    Loads unnormalised band aperiodicites

    Parameters:
    -----------
    save_dir - diretory to load data from

    """
    test_idx = np.load(save_dir + "/val_idx_.npy")
    test_idx = test_idx[k]

    # Infer the dataset size run-time
    example = np.load(save_dir + "/apset_1.npy")
    ap_test = np.zeros((len(test_idx),example.shape[0], example.shape[1]))
    bap_gt_u = np.zeros((len(test_idx),example.shape[0], example.shape[1]))

    for i,idx in enumerate(test_idx):
        ap_test[i,:,:] = np.load(save_dir + "/apset_" +
                                   str(idx) +
                                   ".npy")
    
    scaler_ap = joblib.load(save_dir + "/scaler_ap_.pkl")

    for i in range(len(scaler_ap)):
        bap_gt_u[:,:,i] = scaler_ap[i].inverse_transform(ap_test[:,:,i])

    return bap_gt_u

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """
    
    def __init__(self, options, train, shuffle=True,
                 swap=False,shift=False, channel_idx=[], label="all"):
        """
        Initialisation
        -------------
        The principle that this needs to have access to the concrete
        files anyway, so it needs to be file-specific.
        Thus this should be the entry point for the shape inference
        of the neural network and and it can be fetched via properties
        
        TODO: Howver id subselections needs to be accounted for somehow

        Parameters:
        ------------

        options: Dictionary
        batch_size -
        delay - See delay_signal
        k - which fold to use
        percentage - amount of training data to use
        shuffle - shuffle the IDs
        swap - swap so that Y is regressed against X
        label - which datasets to load, default all
        """
        self.batch_size = options["batch_size"]
        self.delay = options["delay"]
        self.k = options["k"]
        self.shuffle = shuffle
        self.save_dir = options["save_dir"]
        self.percentage = options["percentage"]
        self.shift = shift
        self.swap = swap
        self.channel_idx = channel_idx

        cat_id = np.load(self.save_dir + "/cat_id.npy")
        Xtemp = np.load(self.save_dir + "/dataset_" + str(1) + '.npy')

        if (self.channel_idx == []):
            # Runtime shape inference
            self.in_channel = Xtemp.shape[1]
            self.T = Xtemp.shape[0]
            self.channel_idx = [i for i in range(self.in_channel)]
        else:
            self.in_channel = self.channel_idx.shape[0] 
            self.T = Xtemp.shape[0]
            
        Ytemp = np.load(self.save_dir + "/spset_" + str(1) + '.npy')
        self.out_channel = Ytemp.shape[1]
        
        if (train):
            train_idx = np.load(options["save_dir"] + "/train_idx_.npy")

            print(label)
            if label == "all":
                train_idx = train_idx[self.k]
            else:
                train_idx = np.intersect1d(train_idx[self.k],
                                           cat_id.item()[label])

            size_to_use = int(np.ceil(len(train_idx) * self.percentage))
            print(size_to_use)
            train_idx = train_idx[:size_to_use]
            
            self.list_IDs = train_idx
        else:
            test_idx = np.load(options["save_dir"] + "/val_idx_.npy")

            if label == "all":
                test_idx = test_idx[self.k]
            else:
                
                test_idx = np.intersect1d(test_idx[self.k],
                                           cat_id.item()[label])
                
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
        X, y, w = self.__data_generation(list_IDs_temp)

        return X, y, w

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
        weights = np.zeros((self.batch_size, self.T))
        random_delay = np.random.random_integers(0,100)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            Xtemp = np.load(self.save_dir + "/dataset_" + str(ID) + '.npy')
            X[i,:,:] = Xtemp[:,self.channel_idx]

            # Store class - truncate the end if out_channel is less
            Ytemp = np.load(self.save_dir + "/spset_" + str(ID) + '.npy')
            Y[i,:,:] = Ytemp[:,:self.out_channel]

            # Fetching the signal length for the pure f0
            # and their weighting in the loss function
            f0 = np.load(self.save_dir + "/puref0set_" + str(ID) + '.npy')
            T = len(np.trim_zeros(f0,'b'))
            weights[i,:T+random_delay] = 1
            weights[i,:random_delay] = 0
            
        if self.shift:
            X = delay_signal(X,random_delay)
            Y = delay_signal(Y,random_delay)

        if self.swap:
            return Y, X, weights
        else:
            return X, Y, weights

