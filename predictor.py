

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


something, thing = val_gen.__getitem__(0)
print(type(something))
print(something.shape)

rmse = np.sqrt(np.sum(np.sum((something - ema_test)**2)))
print(rmse)

plt.subplot(1,2,1)
plt.plot(sp_test[0,:,:])
plt.subplot(1,2,2)
plt.plot(thing[0,:,:])
plt.show()


rmse = np.sqrt(np.sum(np.sum((thing - sp_test)**2)))
print(rmse)

mfcc_normalised = mfcc_model.predict_generator(val_gen)
print(mfcc_model.evaluate_generator(val_gen))
print(mfcc_model.evaluate(ema_test,sp_test))
mfcc_normalised_2 = mfcc_model.predict(ema_test).astype(np.float64)

plt.subplot(1,2,1)
plt.plot(mfcc_normalised[0,:,:])
plt.subplot(1,2,2)
plt.plot(mfcc_normalised_2[0,:,:])

plt.show()

mfcc_p_normalised = mfcc_model.predict(path_test).astype(np.float64)

f0 = givenf0_test.astype(np.float64)

mlpg_generated = proc.mlpg_postprocessing(mfcc_normalised,
                                          options["bins_1"],
                                          scaler_sp)
mlpg_p_generated = proc.mlpg_postprocessing(mfcc_p_normalised,
                                            options["bins_1"],
                                            scaler_sp)
print(mlpg_generated.shape)
#print("Loss", np.sqrt(np.sum(np.sum((mfcc_normalised - sp_test)**2))))
#for i in range(len(scaler_sp)):
#    mfcc_normalised[:,:,i] = scaler_sp[i].inverse_transform(mfcc_normalised[:,:,i])

#print("Loss", np.sum(np.sum((mfcc_normalised - sp_test_u)**2)))
#print("MelCD-Delta unnormalised", melcd(mfcc_normalised,sp_test_u))
#print("MelCD", melcd(mlpg_generated,sp_test_u[:,:,:options["bins_1"]]))
    

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
    
    
