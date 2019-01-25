
import numpy as np

from sklearn import preprocessing # For MinMax scale
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time


from keras.models import load_model
from models import model_takuragi,model_gru, model_bigru, model_blstm, model_blstm2, model_blstm2_cpu
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import optimizers

import data_loader
import preprocessing3 as proc
from sklearn.externals import joblib
from sacred import Experiment

from schedules import opt_sched, call_shed

import datetime

from nnmnkwii.metrics import melcd

from schedules import opt_sched

# Fixing the seed for reproducibility

date_at_start = datetime.datetime.now()
date_string = date_at_start.strftime("%y-%b-%d-%H-%m")
ex = Experiment("run_" + date_string)

# TODO: Server on this side
#ex.observers.append(MongoObserver.create(url="localhost:27017"))

# General NN training options, specificities modified inside scope
options = {
    "experiment" : "model_lstm",
    "lr": 0.001, # 0.003 # not assigned in Takuragi paper
    "clip": 5,
    "epochs": 50, #60
    "bins_1": 41,
    "gru": 128,
    "seed": 10,
    "noise": 0.05,
    "delay": 0, # 25 
    "batch_size": 90, #45 # 90 with BLSTM2
    "percentage": 1,
    "k": 0,
    "save_dir": "processed_comb_test",
    "save_analysis": True
}

np.random.seed(options["seed"])

ex.add_config(options)

@ex.automain
def my_main(_config,_run):

    options = _config

    swap = False
    train_gen = data_loader.DataGenerator(options,True,True,swap)
    val_gen = data_loader.DataGenerator(options,False,True,swap)

    options["num_features"] = train_gen.in_channel
    options["out_features"] = train_gen.out_channel
    
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

    if options["save_analysis"]:
        np.save(str(options["percentage"]) +
                "seed" +
                str(options["seed"]) +
                "test" ,learning_curve)
    
    options2 = options
    options2["batch_size"] = 217
    MCD_all = proc.evaluate_validation(model,options2,41)
    print("MCD (dB) (nmkwii)" + str(MCD_all))
    return MCD_all
