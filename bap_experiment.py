
import numpy as np

import soundfile as sf # For reading the wavs
import sounddevice as sd # For playing the wavs

from sklearn import preprocessing # For MinMax scale
import matplotlib.pyplot as plt
import tensorflow as tf
import time

import pyworld as pw

from keras.models import load_model
from models import model_gru
from keras.callbacks import Callback, EarlyStopping, TensorBoard
from keras import optimizers
import preprocessing 
from scipy.io import wavfile
import pysptk as sptk
from sacred import Experiment
import datetime

# Fixing the seed for reproducibility
np.random.seed(2)

class LossHistory(Callback):

    def __init__(self,run,learning_curve):
        self.run = run
        self.i = 0
        
    def on_epoch_end(self,batch,logs):
        self.run.log_scalar("training.loss", logs.get('loss'), self.i)
        self.run.log_scalar("validation.loss", logs.get('val_loss'), self.i)
        self.i = self.i + 1
        

date_at_start = datetime.datetime.now()
date_string = date_at_start.strftime("%y-%b-%d-%H-%m")
ex = Experiment("run_" + date_string)

# TODO: Server on this side
#ex.observers.append(MongoObserver.create(url="localhost:27017"))

# General NN training options, specificities modified inside scope
options = {
    "experiment" : "reproduction_save_attempt",
    "max_input_length" : 2800,
    "num_features": 21,
    "lr": 0.003,
    "epochs": 100,
    "out_features": 41,
    "gru": 128,
    "seed": 10,
    "noise": 0,
    "delay": 25,
    "percentage": 1
}

ex.add_config(options)


@ex.automain
def my_main(_config,_run):

    options = _config

    #preprocessing.preprocess_save()
    # Some hard-coded parameters

    ema_train, ema_test,f0_train, \
    f0_test, sp_train, sp_test, \
    ap_train, ap_test, givenf0_train, \
    givenf0_test, wavdata, \
    scaler_f0, scaler_sp, scaler_ap = preprocessing.load_data(options["delay"],
                                                                           options["percentage"])

    print(wavdata.shape)
    # Extract feature number for convenience
    tb = TensorBoard(log_dir=".logs/stuff" + str(options["noise"]))

    model = model_gru.GRU_Model(options)    
    adam_optimiser = optimizers.Adam(lr=options["lr"])
    model.trainer.compile(optimizer=adam_optimiser, loss="mse")
    model.trainer.fit(ema_train, ap_train, validation_data=(ema_test,ap_test),
                      epochs=options["epochs"],
                      batch_size=10,
                      callbacks=[LossHistory(_run,learning_curve),
                                 #restore_best_weights=True),
                                 tb])
    
    model.trainer.save("checkpoints/model_sp_new.hdf5")

    ap_test_hat = model.trainer.predict(ema_test)

    for k in range(len(scaler_sp)):
        ap_test_hat[:,:,k] = scaler_ap[k].inverse_transform(sp_test_hat[:,:,k])
        ap_test[:,:,k] = scaler_ap[k].inverse_transform(sp_test[:,:,k])

    BAP_all = np.sqrt(np.mean((np.float32(ap_test_hat) - np.float32(ap_test))**2))
    return_string =  "BAP (dB) (for all segments)" +  str(MCD_all)
    return return_string
