
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

import datetime

from nnmnkwii.metrics import melcd

from schedules import opt_sched

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
    "epochs": 100, #60
    "out_features": 82,
    "bins_1": 41,
    "gru": 128,
    "seed": 10,
    "noise": 0,
    "delay": 1, # 25 
    "batch_size": 90, #45 # 90 with BLSTM2
    "percentage": 1,
    "k": 0,
    "total_samples": 2274,
    "save_dir": "processed_comb2_filtered_3"
}

ex.add_config(options)

@ex.automain
def my_main(_config,_run):

    options = _config
    
    # Learning curve storage
    learning_curve = np.zeros((options["epochs"]))
    
    tb = TensorBoard(log_dir="logs/" +
                     date_string +
                     "_" +
                     str(options["noise"]))

    
    mc = ModelCheckpoint("checkpoints/model_sp_comb_lstm_fold_" +
                         str(options["k"]) +
                         ".hdf5",
                         save_best_only=True)

    #model = model_blstm2.LSTM_Model(options)
    model = model_takuragi.GRU_Model(options)
    #optimiser = opt_sched.taguchi_opt()
    optimiser = optimizers.RMSprop(lr=options["lr"],
                                           clipvalue=options["clip"])
    model.trainer.compile(optimizer=optimiser,
                          loss="mse")

    try:
        train_gen = data_loader.DataGenerator(options,True,True)
        val_gen = data_loader.DataGenerator(options,False,True)


        model.trainer.fit_generator(generator=train_gen,
                                    validation_data = val_gen,
                                    epochs=options["epochs"],
                                    callbacks=[tb,
                                               mc])
                                              #LossHistory(_run,learning_curve)])
        
    except KeyboardInterrupt:
        print("Training interrupted")

    model = load_model("checkpoints/model_sp_comb_lstm_fold_" +
                       str(options["k"]) +
                           ".hdf5")


    options2 = options
    options2["batch_size"] = 227
    MCD_all = proc.evaluate_validation(model,options2,41)

    print("MCD (dB) (nmkwii)" + str(MCD_all))
    return MCD_all
