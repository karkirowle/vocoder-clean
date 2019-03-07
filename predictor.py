

import numpy as np
import scipy.signal as signal

from nnmnkwii.metrics import melcd
from nnmnkwii.preprocessing import trim_zeros_frames

import pathology
import preprocessing3 as proc
import audio
import data_loader
import itertools
from keras.models import load_model
from keras_layer_normalization import LayerNormalization

import matplotlib.pyplot as plt

# This should be same as the training options

def synthesis(args):
    options = {
        "delay": 0,
        "percentage": 1,
        "bins_1": 41,
        "bins_2": 1,
        "k": 0,
        "save_dir": "processed_comb_test_4_padded",
        "batch_size": 45,
        "experiment": args.model
    }
    # --------------- LOAD DATA TO TRANSFORM -----------------------------
    options["batch_size"] = 23
    val_gen = data_loader.DataGenerator(options,
                                        False,
                                        False,
                                        False,
                                        args.shift,
                                        label=args.dataset)
    X, Y, _ = val_gen.__getitem__(0)

    N = X.shape[0]
    T = X.shape[1]
    val_gen = None
    # -------------- PATHOLOGICAL SIGNAL PROCESSING ----------------------

    # Method 1: Clip derivative
    #ema_4 =  pathology.derivative_clip(X[4,:,4])

    # Method 2: Upsample the signal
    #path_test = pathology.upsampling(X,total=6500, channels=[4,9])

    # Delay
    #path_test = pathology.delay_signal(X,10,channel_idx=[4,9])

    path_test = pathology.scale_channel(X,2.5,[2,3,4,5])

    #path_test = pathology.add_noise(X,3,[4,5])
    #path_test = pathology.set_channel(X,0,[4,5])
    # Method 3: Zero the channel
    #path_test = pathology.add_noise(X, [4,9])

    #
    # Method 3: 
    #path_test = pathology.derivative_clip(X,0.1,[4,5])

    # -------------- ARTICULATORY TO ACOUSTIC ----------------------------

    mfcc_model = load_model("checkpoints/" + options["experiment"] +
                            str(options["k"]) +
                                ".hdf5",
                            custom_objects =
                            {'LayerNormalization': LayerNormalization} )

    options["batch_size"] = 23
    val_gen = data_loader.DataGenerator(options,
                                        False,
                                        False,
                                        False,
                                        args.shift,
                                        label=args.dataset)

    A, B, _ = val_gen.__getitem__(0)

    mfcc_normalised = mfcc_model.predict_generator(val_gen)
    print(mfcc_normalised)
    print(mfcc_normalised.shape)
    mfcc_p_normalised = mfcc_model.predict(path_test).astype(np.float64)
    print(mfcc_p_normalised.shape)

    f0 = data_loader.load_puref0(options["save_dir"],
                                 options["k"],
                                 args.dataset).astype(np.float64)
    print(f0.shape)
    bap_gt_u = data_loader.load_bap(options["save_dir"],options["k"],
                                    args.dataset)
    scaler_sp = data_loader.load_scalersp(options["save_dir"])

    mlpg_generated = proc.mlpg_postprocessing(mfcc_normalised,
                                              options["bins_1"],
                                              scaler_sp)
    mlpg_p_generated = proc.mlpg_postprocessing(mfcc_p_normalised,
                                                options["bins_1"],
                                                scaler_sp)


    # ---------------------- SPEECH SYNTHESIS ----------------------------

    import sounddevice as sd
    import random

    shuffled = random.sample(range(N), 20)
    for i,id in enumerate(shuffled):

        resynth_length = len(np.trim_zeros(f0[id,:],'b'))

        if resynth_length > 0:
            fname = "sounds5/normal/" + str(i) + ".wav"
            print(fname)
            sound1 = audio.save_resynth(fname,f0[id,:resynth_length],
                                mlpg_generated[id,:resynth_length,:],
                                bap_gt_u[id,:resynth_length,:],
                                fs=16000,
                                an=5)
            fname = "sounds5/pathological/" + str(i) + ".wav"

            sound2 = audio.save_resynth(fname,f0[id,:resynth_length],
                                mlpg_p_generated[id,:resynth_length,:],
                                bap_gt_u[id,:resynth_length,:],
                                fs=16000,
                                an=5)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--shift', action="store_true")
    parser.add_argument("--dataset", choices=["all",
                                              "mngu0",
                                              "male",
                                              "female",
                                              "d3", "d4",
                                              "d5", "d6",
                                              "d7", "d8",
                                              "d9", "d10",
                                              "d11", "d12",
                                              "d13"])
    args = parser.parse_args()
    synthesis(args)
