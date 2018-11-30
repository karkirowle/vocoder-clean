

from keras.models import load_model
import preprocessing2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
options = {
    "id": 4,
    "delay": 25,
    "percentage": 1
    }

ema_test, sp_test, ap_test, givenf0_test, scaler_f0, scaler_sp, \
        scaler_ap = preprocessing2.load_test(
            options["delay"],
            options["percentage"])





id = 2


# Method 1: Clip derivative
#t = 0.01
#ema_4_diff = np.diff(ema_test[id,:,4])
#ema_4_diff[ema_4_diff > t] = t
#ema_4 = np.cumsum(np.insert(ema_4_diff,0,ema_test[id,0,4]))


#ema_9_diff = np.diff(ema_test[id,:,9])
#ema_9_diff[ema_9_diff > t] = t
#ema_9 = np.cumsum(np.insert(ema_9_diff,0,ema_test[id,0,9]))



# Method 2: Stay at level 0  

#ema_test[id,:,4] = ema_test[id,0,4] * np.ones((2800))
#ema_test[id,:,9] = ema_test[id,0,9] * np.ones((2800))

# Method 3: Upsample the signal
#t = 6500
#ema_test[id,:,4] = signal.resample(ema_test[id,:,4],t)[:2800]
#ema_test[id,:,9] = signal.resample(ema_test[id,:,9],t)[:2800]


# From checkpoints load the keras models
mfcc_model = load_model("checkpoints/model_sp_new.hdf5")
bap_model = load_model("checkpoints/model_bap_new.hdf5")



mfcc_unnormalised = mfcc_model.predict(ema_test).astype(np.float64)
#bap_unnormalised = bap_model.predict(ema_test).astype(np.float64)
f0 = givenf0_test.astype(np.float64)
fs = 16000

mfcc_normalised = np.copy(mfcc_unnormalised)
mfcc_gt_normalised = np.copy(mfcc_unnormalised)
#bap_normalised = np.copy(bap_unnormalised)
bap_gt_normalised = np.copy(ap_test)

print(mfcc_normalised.shape)
for i in range(len(scaler_sp)):
    mfcc_normalised[:,:,i] = scaler_sp[i].inverse_transform(mfcc_unnormalised[:,:,i])
    mfcc_gt_normalised[:,:,i] = scaler_sp[i].inverse_transform(sp_test[:,:,i])

for i in range(len(scaler_ap)):
    #bap_normalised[:,:,i] = scaler_ap[i].inverse_transform(bap_unnormalised[:,:,i])
    bap_gt_normalised[:,:,i] = scaler_ap[i].inverse_transform(ap_test[:,:,i])
    
    
#plt.plot(f0[0,:])
#plt.show()
plt.imshow(mfcc_normalised[0,:,:],aspect="auto")
plt.show()
plt.imshow(bap_gt_normalised[0,:,:],aspect="auto")
plt.show()

for id in range(100):
    preprocessing2.debug_resynth(f0[id,:],
                                mfcc_normalised[id,:],
                                bap_gt_normalised[id,:],fs)
