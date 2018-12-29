

from keras.models import load_model
import preprocessing3
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from nnmnkwii.metrics import melcd
from nnmnkwii.paramgen import mlpg
from nnmnkwii.preprocessing import trim_zeros_frames
import seaborn as sns


from keras_layer_normalization import LayerNormalization
options = {
    "id": 4,
    "delay": 1,
    "percentage": 1
    }

ema_test, sp_test, ap_test, givenf0_test, scaler_f0, scaler_sp, \
        scaler_ap = preprocessing3.load_test(
            options["delay"],
            options["percentage"])





# Method 1: Clip derivative

id = 4
sns.set_style("darkgrid")
plt.subplot(1,2,1)

plt.plot(ema_test[id,:,4])
plt.title("Original EMA signal for tongue tip")
plt.ylim([-2, 3.5])
plt.xlabel("time [s]")
plt.ylabel("normalised x position")
t = 0.1
ema_4_diff = np.diff(ema_test[id,:,4])
ema_4_diff[ema_4_diff > t] = t
ema_4_diff[ema_4_diff < -t] = -t
ema_4 = np.cumsum(np.insert(ema_4_diff,0,ema_test[id,0,4]))
plt.subplot(1,2,2)
plt.plot(ema_4)
plt.title("Modified EMA signal for tongue tip")
plt.ylim([-2, 3.5])
plt.xlabel("time [s]")
plt.ylabel("normalised x position")
plt.show()
#ema_9_diff = np.diff(ema_test[id,:,9])
#ema_9_diff[ema_9_diff > t] = t
#ema_9 = np.cumsum(np.insert(ema_9_diff,0,ema_test[id,0,9]))



# Method 2: Stay at level 0  

#ema_test[id,:,4] = ema_test[id,0,4] * np.ones((1000))
#ema_test[id,:,9] = ema_test[id,0,9] * np.ones((1000))
path_test = np.copy(ema_test)

# Method 3: Upsample the signal
t = 6500
path_test[:,:1000,4] = signal.resample(ema_test[:,:,4],t,axis=1)[:,:1000]
path_test[:,:1000,9] = signal.resample(ema_test[:,:,9],t,axis=1)[:,:1000]

mfcc_model = load_model("checkpoints/model_sp_comb_lstm_noise2.hdf5", custom_objects =
                        {'LayerNormalization': LayerNormalization} )
#mfcc_model = load_model("checkpoints/presentation_model.hdf5", custom_objects =
#                        {'LayerNormalization': LayerNormalization} )

# Select feature ids
idx = [0,1,2,3,4,5,6,7,10,11,12,13]
ema_test = ema_test[:,:,:]
path_test = path_test[:,:,:]

# Feedforward pass
mfcc_unnormalised = mfcc_model.predict(ema_test).astype(np.float64)
mfcc_p_unnormalised = mfcc_model.predict(path_test).astype(np.float64)

f0 = givenf0_test.astype(np.float64)
fs = 16000

mfcc_normalised = np.copy(mfcc_unnormalised)
mfcc_p_normalised = np.copy(mfcc_unnormalised)
mfcc_gt_normalised = np.copy(mfcc_unnormalised)
bap_gt_normalised = np.zeros((mfcc_normalised.shape[0],1000,1))

windows = [
    (0, 0, np.array([1.0])),            # static
    (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
]


mlpg_generated = np.zeros((mfcc_normalised.shape[0],
                          mfcc_normalised.shape[1],
                           41))
mlpg_p_generated = np.zeros((mfcc_normalised.shape[0],
                          mfcc_normalised.shape[1],
                           41))

# Maximum Likelihood Parameter Generation 
for id in range(mfcc_normalised.shape[0]):
    mlpg_generated[id,:,:] = mlpg(mfcc_unnormalised[id,:,:], np.ones((mfcc_normalised.shape[1],
                                                            mfcc_normalised.shape[2])),
                          windows)
    mlpg_p_generated[id,:,:] = mlpg(mfcc_p_unnormalised[id,:,:],
                                  np.ones((mfcc_normalised.shape[1],
                                           mfcc_normalised.shape[2])),
                                           windows)

for i in range(len(scaler_sp)):
    mlpg_generated[:,:,i] = scaler_sp[i].inverse_transform(mlpg_generated[:,:,i])
    mlpg_p_generated[:,:,i] = scaler_sp[i].inverse_transform(mlpg_p_generated[:,:,i])
    mfcc_gt_normalised[:,:,i] = scaler_sp[i].inverse_transform(sp_test[:,:,i])
print(melcd(mlpg_generated,mfcc_gt_normalised[:,:,:41]))
    
for i in range(len(scaler_ap)):
    bap_gt_normalised[:,:,i] = scaler_ap[i].inverse_transform(ap_test[:,:,i])
    
for id in range(100):
    resynth_length = len(np.trim_zeros(f0[id,:],'b'))
    print(resynth_length)

    fname = "sounds3/" + str(id)
    print(fname)

    preprocessing3.debug_resynth(f0[id,:resynth_length],
                                 mlpg_generated[id,:resynth_length],
                                 bap_gt_normalised[id,:resynth_length],
                                 fs, an=5)
    preprocessing3.debug_resynth(f0[id,:resynth_length],
                                 mlpg_p_generated[id,:resynth_length],
                                 bap_gt_normalised[id,:resynth_length],
                                 fs, an=5)
#    preprocessing3.save_resynth(fname + "_gt.wav",
#                                f0[id,:resynth_length],
#                                 mfcc_gt_normalised[id,:resynth_length],
#                                 bap_gt_normalised[id,:resynth_length],fs,an=5)
#    preprocessing3.save_resynth(fname + "_rs.wav",
#                                f0[id,:resynth_length],
#                                mlpg_generated[id,:resynth_length],
#                                 bap_gt_normalised[id,:resynth_length],fs,an=5)
#    
#    preprocessing3.save_resynth(fname + "_p.wav",
#                                f0[id,:resynth_length],
#                                mlpg_p_generated[id,:resynth_length],
#                                 bap_gt_normalised[id,:resynth_length],fs,an=5)
    
