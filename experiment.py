
import numpy as np

from sklearn import preprocessing # For MinMax scale
import matplotlib.pyplot as plt
import tensorflow as tf
import time


from keras.models import load_model
from models import model_takuragi,model_gru, model_bigru, model_blstm, model_blstm2
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import optimizers

import data_loader

from sacred import Experiment

import datetime

from nnmnkwii.metrics import melcd

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
    "num_features": 15,
    "lr": 0.01, # 0.003 # not assigned in Takuragi paper
    "clip": 5,
    "epochs": 200,
    "out_features": 82,
    "gru": 128,
    "seed": 10,
    "noise": 0,
    "delay": 1, # 25 
    "batch_size": 45, #90
    "percentage": 1
}

ex.add_config(options)

@ex.automain
def my_main(_config,_run):

    options = _config

    # Learning curve storage
    learning_curve = np.zeros((options["epochs"]))
    
    # Some hard-coded parameters

    ema_train, ema_test, \
        sp_train, sp_test, \
        scaler_sp = data_loader.load_data(options["delay"],
                                   options["percentage"])

    # Extract feature number for convenience

    tb = TensorBoard(log_dir="logs/" +
                     date_string +
                     "_" +
                     str(options["noise"]))

    mc = ModelCheckpoint("checkpoints/model_sp_comb_lstm_d.hdf5" ,
                         save_best_only=True)

    model = model_blstm2.LSTM_Model(options)    

    rmsprop_optimiser = optimizers.RMSprop(lr=options["lr"],
                                           clipvalue=options["clip"])
    model.trainer.compile(optimizer=rmsprop_optimiser, loss="mse")

    try:
        model.trainer.fit(ema_train,
                          sp_train,
                          validation_data=(ema_test,sp_test),
                      epochs=options["epochs"],
                      batch_size=options["batch_size"],
                      callbacks=[LossHistory(_run,learning_curve),
                                 mc,
                                 tb])
    except KeyboardInterrupt:
        print("Training interrupted")

    sp_test_hat = model.trainer.predict(ema_test)

    for k in range(len(scaler_sp)):
        sp_test_hat[:,:,k] = scaler_sp[k].inverse_transform(sp_test_hat[:,:,k])
        sp_test[:,:,k] = scaler_sp[k].inverse_transform(sp_test[:,:,k])

    MCD_all = str(melcd(sp_test_hat,sp_test))
    print("MCD (dB) (nmkwii)" + MCD_all)
    return MCD_all
