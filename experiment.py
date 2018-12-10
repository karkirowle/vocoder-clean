
import numpy as np

import soundfile as sf # For reading the wavs
import sounddevice as sd # For playing the wavs

from sklearn import preprocessing # For MinMax scale
import matplotlib.pyplot as plt
import tensorflow as tf
import time

import pyworld as pw

from keras.models import load_model
from models import model_takuragi,model_gru, model_bigru, model_blstm
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import optimizers
from keras.losses import mean_squared_error
import preprocessing3
from scipy.io import wavfile
import pysptk as sptk
from sacred import Experiment
import datetime

from nnmnkwii.paramgen import mlpg, unit_variance_mlpg_matrix
from nnmnkwii.metrics import melcd
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

    ema_train, ema_test,f0_train, \
    f0_test, sp_train, sp_test, \
    ap_train, ap_test, givenf0_train, \
    givenf0_test, wavdata, \
    scaler_f0, scaler_sp, scaler_ap = preprocessing3.load_data(options["delay"],
                                                               options["percentage"])
    print(ema_train.shape)
    #idx = [0,1,2,3,4,5,6,7,10,11,12,13]
    ema_train = ema_train
    ema_test = ema_test
    
    print(sp_train.shape)
    # Extract feature number for convenience
    tb = TensorBoard(log_dir="logs/" + date_string + "_" + str(options["noise"]))
    #es = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=100,
    #                   restore_best_weights=True)
    mc = ModelCheckpoint("checkpoints/model_sp_takuragi.hdf5" , save_best_only=True)
    model = model_blstm.LSTM_Model(options)    
    adam_optimiser = optimizers.Adam(lr=options["lr"])
    sgd_optimiser = optimizers.SGD(lr=options["lr"])
    rmsprop_optimiser = optimizers.RMSprop(lr=options["lr"],clipvalue=5)

    def mmg_loss(x_true,x_pred):
        windows = [
            (0, 0, np.array([1.0])),            # static
            (1, 1, np.array([-0.5, 0.0, 0.5]))] # delta
        nnmnkwii.autograd.MLPG(np.zeros((100,1)))
     #   static_pred = np.zeros((options["batch_size"],1000,82))
      #  static_true = np.zeros((options["batch_size"],1000,82))
       # unit_variance_mlpg_matrix(windows
        #for i in range(options["batch_size"]):
         #   dummy_var = np.ones((x_pred.shape[1],82))
          #  print(dummy_var.shape)
           # print(x_pred.shape)
            #static_pred[i,:,:] = mlpg(x_pred[i,:,:],dummy_var,windows)
            #static_true[i,:,:] = mlpg(x_true[i,:,:],dummy_var,windows)
       # return mean_squared_error(static_true, static_pred)

    
    model.trainer.compile(optimizer=rmsprop_optimiser, loss="mse")
    try:
        model.trainer.fit(ema_train, sp_train, validation_data=(ema_test,sp_test),
                      epochs=options["epochs"],
                      batch_size=options["batch_size"],
                      callbacks=[LossHistory(_run,learning_curve),
                                 mc,
                                 tb])
    except KeyboardInterrupt:
        print("Training interrupted")
    #model.trainer.save("checkpoints/model_sp_comb.hdf5")

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
    print("MCD (dB) (nmkwii)" + str(melcd(sp_test_hat,sp_test)))
    return return_string
