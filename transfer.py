import numpy as np

from sklearn import preprocessing # For MinMax scale
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time


from keras.models import load_model
from models import model_blstm3, transfer_blstm, model_lstm_conv, model_conv, model_bigru
from models import model_takuragi, model_zhengcheng, model_bigru_proc
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import optimizers
from keras_layer_normalization import LayerNormalization

import data_loader
import preprocessing3 as proc
from sklearn.externals import joblib
from sacred import Experiment

from schedules import opt_sched, call_shed

import datetime

from nnmnkwii.metrics import melcd

from schedules import opt_sched

date_at_start = datetime.datetime.now()
date_string = date_at_start.strftime("%y-%b-%d-%H-%m")
ex = Experiment("run_" + date_string)

@ex.main
def my_main(_config,_run):

    options = _config

    swap = False

    if args.f0:
        channel_idx = np.array([0,1,2,3,4,5,6,7,10,11,12,13,14])
    else:
        channel_idx = np.array([0,1,2,3,4,5,6,7,10,11,12,13])

    # ----------- STEP 1 - Train a baseline on mngu0 -----------------
    options["batch_size"] = 50
    train_gen = data_loader.DataGenerator(options,True,True,swap,
                                          args.shift,label="mngu0")
    val_gen = data_loader.DataGenerator(options,False,True,swap,
                                        args.shift,label="mngu0")
    
    options["num_features"] = train_gen.in_channel
    options["out_features"] = train_gen.out_channel
    
    # Learning curve storage
    learning_curve = np.zeros((options["epochs"]))

    model = model_bigru_proc.GRU_Model(options)
        
    optimiser = optimizers.Adam(lr=options["lr"])
    model.trainer.compile(optimizer=optimiser,
                          loss="mse",
                          sample_weight_mode="temporal")

    cb = call_shed.fetch_callbacks(options,_run,learning_curve)

    if args.train:
        try:
            model.trainer.fit_generator(generator=train_gen,
                                        validation_data = val_gen,
                                        epochs=options["epochs"],
                                        callbacks = cb)

        except KeyboardInterrupt:
            print("Training interrupted")


        model.trainer = load_model("checkpoints/" +
                                   options["experiment"] +
                                   str(options["k"]) +
                                   ".hdf5")

        options2 = options
        options2["batch_size"] = 30
        MCD_all = proc.evaluate_validation(model.trainer,options2,41,"mngu0")
        print("MCD (dB) (nmkwii)" + str(MCD_all))

        # ----------- STEP 2 - Train on only chosen dataset ----------
        # PRESET - Training on special data determined externally
        options["batch_size"] = 30
        train_gen = data_loader.DataGenerator(options,True,True,swap,
                                              args.shift,
                                              label=args.dataset)
        val_gen = data_loader.DataGenerator(options,False,True,swap,
                                            args.shift,
                                            label=args.dataset)
        
        for layer in model.transfer.layers[-5:-1]:
                print(layer)
                layer.trainable = False
        model.transfer.compile(optimizer=optimiser,
                    loss="mse",
                    sample_weight_mode="temporal")
        try:
            model.transfer.fit_generator(generator=train_gen,
                                        validation_data = val_gen,
                                        epochs=options["epochs"],
                                callbacks = cb)
            print("ide eljut")
        except KeyboardInterrupt:
            print("Training interrupted")

    print("ide eljut")

    model.transfer = load_model("checkpoints/" + options["experiment"] +
                       str(options["k"]) +
                           ".hdf5")

    if options["save_analysis"]:
        np.save(str(options["percentage"]) +
                "seed" +
                str(options["seed"]) +
                "test" ,learning_curve)

    options2 = options
    options2["batch_size"] = 30
    MCD_all = proc.evaluate_validation(model.transfer,options2,41,args.dataset)
    print("MCD (dB) (nmkwii)" + str(MCD_all))
    return MCD_all

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dataset", choices=["all", "mngu0","male",
                                              "female", "d3", "d4",
                                              "d5", "d6", "d7", "d8",
                                              "d9", "d10", "d11", "d12"])
    parser.add_argument('--batch', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--shift', action="store_true")
    parser.add_argument('--experiment', type=str)
    args = parser.parse_args()


    # TODO: Server on this side
    #ex.observers.append(MongoObserver.create(url="localhost:27017"))

    # General NN training options, specificities modified inside scope
    options = {
        "experiment" : args.experiment,
        "lr": args.lr, # 0.003 # not assigned in Takuragi paper
        "clip": 5,
        "epochs": 50, #60
        "bins_1": 41,
        "gru": 128,
        "seed": 25, #10
        "noise": 0.05,
        "delay": 0, # 25 
        "batch_size": args.batch, #45 # 90 with BLSTM2
        "percentage": 1,
        "k": 0,
        "save_dir": "processed_comb_test_3_padded",
        "save_analysis": True
    }

    # Fixing the seed for reproducibility
    np.random.seed(options["seed"])

    ex.add_config(options)

    ex.run()
