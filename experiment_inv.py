
import numpy as np

from sklearn import preprocessing # For MinMax scale
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time


from keras.models import load_model
from models import model_blstm2
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import optimizers

import data_loader
import preprocessing3 as proc
from schedules import opt_sched, call_shed
from sklearn.externals import joblib
from sacred import Experiment

import datetime



# Fixing the seed for reproducibility
np.random.seed(2)



date_at_start = datetime.datetime.now()
date_string = date_at_start.strftime("%y-%b-%d-%H-%m")
ex = Experiment("run_" + date_string)

# TODO: Server on this side
#ex.observers.append(MongoObserver.create(url="localhost:27017"))

# General NN training options, specificities modified inside scope
options = {
    "experiment" : "model_inversion",
    "lr": 0.01, # 0.003 # not assigned in Takuragi paper
    "clip": 5,
    "epochs": 100, #60
    "gru": 128,
    "seed": 10,
    "noise": 0.001,
    "delay": 0, # 25 
    "batch_size": 90, #45 # 90 with BLSTM2
    "k": 0,
    "save_dir": "processed_comb2_filtered_3"
}

ex.add_config(options)

import matplotlib.pyplot as plt
@ex.automain
def my_main(_config,_run):

    options = _config

    swap = True
    train_gen = data_loader.DataGenerator(options,True,True,swap)
    val_gen = data_loader.DataGenerator(options,False,True,swap)

    Y, X = train_gen.__getitem__(1)

    # Becausse swapped
    options["num_features"] = train_gen.out_channel
    options["out_features"] = train_gen.in_channel

    # Learning curve storage
    learning_curve = np.zeros((options["epochs"]))
    
    model = model_blstm2.LSTM_Model(options)
    optimiser = optimizers.Adam(lr=options["lr"])
    model.trainer.compile(optimizer=optimiser,
                          loss="mse")

    cb = call_shed.fetch_callbacks(options,_run,learning_curve)

    try:

        model.trainer.fit_generator(generator=train_gen,
                                    validation_data = val_gen,
                                    epochs=options["epochs"],
                                    callbacks=cb)
        
    except KeyboardInterrupt:
        print("Training interrupted")

    model = load_model("checkpoints/" + options["experiment"] +
                       str(options["k"]) +
                           ".hdf5")

    sp_test_hat = model.predict_generator(val_gen)
    _, sp_test = val_gen.__getitem__(0)

    plt.subplot(1,2,1)
    plt.plot(sp_test_hat[0,:,8])
    plt.subplot(1,2,2)
    plt.plot(sp_test[0,:,8])
    plt.show()
    return 1
