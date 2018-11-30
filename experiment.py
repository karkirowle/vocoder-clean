
import numpy as np

import soundfile as sf # For reading the wavs
import sounddevice as sd # For playing the wavs

from sklearn import preprocessing # For MinMax scale
import matplotlib.pyplot as plt
import tensorflow as tf
import time

import pyworld as pw

from keras.models import load_model
from models import model_gru, model_bigru, model_blstm
from keras.callbacks import Callback, EarlyStopping, TensorBoard
from keras import optimizers
import preprocessing3
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
        self.learning_curve = learning_curve
        
    def on_epoch_end(self,batch,logs):
        self.run.log_scalar("training.loss", logs.get('loss'), self.i)
        self.run.log_scalar("validation.loss", logs.get('val_loss'), self.i)
        self.learning_curve[self.i] = logs.get('val_loss')
        self.i = self.i + 1
        

date_at_start = datetime.datetime.now()
date_string = date_at_start.strftime("%y-%b-%d-%H-%m")
ex = Experiment("run_" + date_string)

# TODO: Server on this side
#ex.observers.append(MongoObserver.create(url="localhost:27017"))

# General NN training options, specificities modified inside scope
options = {
    "experiment" : "reproduction_save_attempt",
    "max_input_length" : 1000,
    "num_features": 12,
    "lr": 0.01, # 0.003
    "epochs": 60,
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

    # Learning curve storage
    learning_curve = np.zeros((options["epochs"]))
    
    # Some hard-coded parameters

    ema_train, ema_test,f0_train, \
    f0_test, sp_train, sp_test, \
    ap_train, ap_test, givenf0_train, \
    givenf0_test, wavdata, \
    scaler_f0, scaler_sp, scaler_ap = preprocessing3.load_data(options["delay"],
                                                               options["percentage"])

    idx = [0,1,2,3,4,5,6,7,10,11,12,13]
    ema_train = ema_train[:,:,idx]
    ema_test = ema_test[:,:,idx]
    
    print(wavdata.shape)
    print(sp_train.shape)
    # Extract feature number for convenience
    tb = TensorBoard(log_dir="logs/" + date_string + "_" + str(options["noise"]))
    es = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=100,
                       restore_best_weights=True)
    
    model = model_blstm.LSTM_Model(options)    
    adam_optimiser = optimizers.Adam(lr=options["lr"])
    sgd_optimiser = optimizers.SGD(lr=options["lr"])
    model.trainer.compile(optimizer=adam_optimiser, loss="mse")
    model.trainer.fit(ema_train, sp_train, validation_data=(ema_test,sp_test),
                      epochs=options["epochs"],
                      batch_size=10,
                      callbacks=[LossHistory(_run,learning_curve),
                                 #es,
                                 tb])
    
    model.trainer.save("checkpoints/model_sp_new.hdf5")

#    filename = "analysis/retraining/" + str(options["percentage"]) + "_test"
#    np.savetxt(filename, learning_curve, delimiter=','

    sp_test_hat = model.trainer.predict(ema_test)

    for k in range(len(scaler_sp)):
        sp_test_hat[:,:,k] = scaler_sp[k].inverse_transform(sp_test_hat[:,:,k])
        sp_test[:,:,k] = scaler_sp[k].inverse_transform(sp_test[:,:,k])

    MCD_all = np.sum(np.sqrt(np.sum( (sp_test_hat - sp_test)**2,axis=2)),axis=1)
    MCD_all = (10 * np.sqrt(2))/(sp_test.shape[1] * np.log(10)) * MCD_all
    MCD_all = np.mean(MCD_all)
    return_string =  "MCD (dB) (for all segments)" +  str(MCD_all)
    return return_string
