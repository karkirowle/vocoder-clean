
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

from schedules import opt_sched, call_shed

import datetime

from nnmnkwii.metrics import melcd

from schedules import opt_sched

from hyperas import optim
from hyperas.distributions import choice,uniform
from hyperopt import Trials, STATUS_OK, tpe

from keras.layers import Input, Dense, GaussianNoise, Bidirectional, CuDNNLSTM
from keras.models import Model
from keras.optimizers import Adam

def data():
    options = {
        "delay": 0,
        "batch_size": 90,
        "k": 0,
        "save_dir": "processed_comb_test",
        "percentage": 1
    }
    swap = False
    
    train_gen = data_loader.DataGenerator(options,True,True,swap)
    val_gen = data_loader.DataGenerator(options,False,True,swap)
    return train_gen, val_gen

def create_model(train_gen, val_gen):
    options = {}
    options["num_features"] = train_gen.in_channel
    options["out_features"] = train_gen.out_channel


    inputs = Input(shape=(None,options["num_features"]))
    noise = GaussianNoise({{uniform(0,0.1)}})(inputs)

    # LSTM layers share number of hidden layer parameter
    gru_1a = Bidirectional(CuDNNLSTM({{choice([128,256,512,1024])}},
                                    return_sequences=True))(noise)
    gru_2a = Bidirectional(CuDNNLSTM({{choice([128,256,512,1024])}},
                                    return_sequences=True))(gru_1a)
    gru_3a = Bidirectional(CuDNNLSTM({{choice([128,256,512,1024])}},
                                    return_sequences=True))(gru_2a)
    gru_4a = Bidirectional(CuDNNLSTM({{choice([128,256,512,1024])}},
                                    return_sequences=True))(gru_3a)

    # Densex
    dense = Dense(options["out_features"])(gru_4a)
    model = Model(inputs, dense)


    optimiser = optimizers.Adam(lr={{uniform(0,0.1)}})

    model.compile(optimizer=optimiser,
                          loss="mse")


    try:
        model.fit_generator(generator=train_gen,
                            validation_data = val_gen,
                            epochs=100)
    except KeyboardInterrupt:
        print("Training interrupted")

    model = load_model("checkpoints/" + options["experiment"] +
                       str(options["k"]) +
                           ".hdf5")
    options2 = options
    options2["batch_size"] = 227

    MCD_all = proc.evaluate_validation(model,options2,41)

    return {'loss': MCD_all, 'status': STATUS_OK, 'model': model}

        
if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())

    print("Test set is for the weak")
    print(best_run)
    
