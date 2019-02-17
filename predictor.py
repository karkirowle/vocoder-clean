

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
    "delay": 0,
    "save_dir": "processed_comb_test_3",
    "batch_size": 45,
    "experiment": "model_lstm0"
}


# --------------- LOAD DATA TO TRANSFORM -----------------------------
options["batch_size"] = 217
val_gen = data_loader.DataGenerator(options,False,False)
X, Y = val_gen.__getitem__(0)

N = X.shape[0]
T = X.shape[1]


# -------------- PATHOLOGICAL SIGNAL PROCESSING ----------------------

# Method 1: Clip derivative
#ema_4 =  pathology.derivative_clip(X[4,:,4])

# Method 2: Upsample the signal
path_test = pathology.upsampling(X,total=6500, channels=[4,9])

# Method 3: 
#path_test = pathology.derivative_clip_2(X,0.1,[4,5])

# -------------- ARTICULATORY TO ACOUSTIC ----------------------------

mfcc_model = load_model("checkpoints/" + options["experiment"] + ".hdf5",
                        custom_objects =
                        {'LayerNormalization': LayerNormalization} )

val_gen = data_loader.DataGenerator(options,False,False)


mfcc_normalised = mfcc_model.predict_generator(val_gen)



mfcc_p_normalised = mfcc_model.predict(path_test).astype(np.float64)

f0 = data_loader.load_puref0(options["save_dir"],
                             options["k"]).astype(np.float64)
bap_gt_u = data_loader.load_bap(options["save_dir"],options["k"])
scaler_sp = data_loader.load_scalersp(options["save_dir"])

mlpg_generated = proc.mlpg_postprocessing(mfcc_normalised,
                                          options["bins_1"],
                                          scaler_sp)
mlpg_p_generated = proc.mlpg_postprocessing(mfcc_p_normalised,
                                            options["bins_1"],
                                            scaler_sp)


# ---------------------- SPEECH SYNTHESIS ----------------------------

import sounddevice as sd
for id in range(N):

    resynth_length = len(np.trim_zeros(f0[id,:],'b'))

    if resynth_length > 0:
        fname = "sounds3/" + str(id)
        print(fname)

        sound1 = audio.debug_resynth(f0[id,:resynth_length],
                            mlpg_generated[id,:resynth_length,:],
                            bap_gt_u[id,:resynth_length,:],
                            fs=16000,
                            an=5)

        sound2 = audio.debug_resynth(f0[id,:resynth_length],
                            mlpg_p_generated[id,:resynth_length,:],
                            bap_gt_u[id,:resynth_length,:],
                            fs=16000,
                            an=5)


        if id == 1:
            sd.play(sound1[14000:22000])
            sd.wait()
            sd.play(sound2[14000:22000])
            sd.wait()

        plt.plot(sound1)
        plt.plot(sound2)
        plt.show()
    
