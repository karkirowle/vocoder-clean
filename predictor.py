

import numpy as np
import scipy.signal as signal

from nnmnkwii.metrics import melcd
from nnmnkwii.preprocessing import trim_zeros_frames

import pathology
import preprocessing3 as proc
import audio
import data_loader

from keras.models import load_model
from keras_layer_normalization import LayerNormalization

import matplotlib.pyplot as plt

# This should be same as the training options
options = {
    "delay": 1,
    "percentage": 1,
    "bins_1": 41,
    "bins_2": 1,
    "k": 0,
    "num_features": 15,
    "out_features": 82,
    "delay": 1,
    "save_dir": "processed_comb2_filtered_2",
    "batch_size": 45
}

ema_test, sp_test, ap_test, givenf0_test, scaler_sp, \
        scaler_ap, sp_test_u, bap_gt_u = data_loader.load_test(
            options["delay"],
            options["percentage"],
            options["save_dir"],
            options["k"])

N = ema_test.shape[0]
T = ema_test.shape[1]

options["batch_size"] = N
# -------------- PATHOLOGICAL SIGNAL PROCESSING ----------------------

# Method 1: Clip derivative
ema_4 =  pathology.derivative_clip(ema_test[4,:,4])

# Method 2: Upsample the signal
path_test = pathology.upsampling(ema_test,total=6500, channels=[4,9])


# -------------- ARTICULATORY TO ACOUSTIC ----------------------------

mfcc_model = load_model("checkpoints/model_sp_comb_lstm_fold_0.hdf5",
                        custom_objects =
                        {'LayerNormalization': LayerNormalization} )

val_gen = data_loader.DataGenerator(options,False,False)


mfcc_normalised = mfcc_model.predict_generator(val_gen)



mfcc_p_normalised = mfcc_model.predict(path_test).astype(np.float64)

f0 = givenf0_test.astype(np.float64)

mlpg_generated = proc.mlpg_postprocessing(mfcc_normalised,
                                          options["bins_1"],
                                          scaler_sp)
mlpg_p_generated = proc.mlpg_postprocessing(mfcc_p_normalised,
                                            options["bins_1"],
                                            scaler_sp)
print(mlpg_generated.shape)

# ---------------------- SPEECH SYNTHESIS ----------------------------

for id in range(N):

    resynth_length = len(np.trim_zeros(f0[id,:],'b'))

    fname = "sounds3/" + str(id)
    print(fname)

    audio.debug_resynth(f0[id,:resynth_length],
                        mlpg_generated[id,:resynth_length],
                        bap_gt_u[id,:resynth_length],
                        fs=16000,
                        an=5)

    audio.debug_resynth(f0[id,:resynth_length],
                        mlpg_p_generated[id,:resynth_length],
                        bap_gt_u[id,:resynth_length],
                        fs=16000,
                        an=5)
    
    
