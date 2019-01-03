
import numpy as np

from sklearn import preprocessing # For MinMax scale
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time


from keras.models import load_model
from models import model_takuragi,model_gru, model_bigru, model_blstm, model_blstm2
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import optimizers

import data_loader
import preprocessing3 as proc
from sklearn.externals import joblib
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
    "epochs": 1, #60
    "out_features": 82,
    "gru": 128,
    "seed": 10,
    "noise": 0,
    "delay": 1, # 25 
    "batch_size": 45, #90
    "percentage": 1,
    "k": 0,
    "total_samples": 2274,
    "save_dir": "processed_comb2_filtered_2"
}

ex.add_config(options)

@ex.main
def my_main(_config,_run):

    options = _config
    total_samples = options["total_samples"]
    train_size = int(np.ceil(total_samples * 0.8))
    val_size = total_samples - train_size
    
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

    model = model_blstm2.LSTM_Model(options)    

    rmsprop_optimiser = optimizers.RMSprop(lr=options["lr"],
                                           clipvalue=options["clip"])
    model.trainer.compile(optimizer=rmsprop_optimiser, loss="mse")

    try:
        train_gen = data_loader.DataGenerator(options,True,True)
        val_gen = data_loader.DataGenerator(options,False,True)

        model.trainer.fit_generator(generator=train_gen,
                                    validation_data = val_gen,
                                    epochs=options["epochs"],
                                    callbacks=[tb,
                                              mc,
                                              LossHistory(_run,learning_curve)])
        
    except KeyboardInterrupt:
        print("Training interrupted")

    model = load_model("checkpoints/model_sp_comb_lstm_fold_" +
                       str(options["k"]) +
                           ".hdf5")


    options2 = options
    options2["batch_size"] = 227
    val_gen = data_loader.DataGenerator(options2,False,False)
    sp_test_hat = model.predict_generator(val_gen)
    _, sp_test = val_gen.__getitem__(0)

    print(sp_test.shape)
    scaler_sp = joblib.load(options["save_dir"] + '/scaler_sp_.pkl')

    # Perform MLPG
    mlpg_generated = proc.mlpg_postprocessing(sp_test_hat,
                                          41,
                                          scaler_sp)

    sp_test_u = np.copy(sp_test_hat)
    for i in range(len(scaler_sp)):
        sp_test_u[:,:,i] = scaler_sp[i].inverse_transform(sp_test[:,:,i])
    
    MCD_all = str(melcd(mlpg_generated,sp_test_u[:,:,:41]))

    print("MCD (dB) (nmkwii)" + MCD_all)
    return MCD_all
