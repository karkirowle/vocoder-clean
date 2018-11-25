

from keras.models import load_model
import preprocessing
import numpy as np
import matplotlib.pyplot as plt
options = {
    "id": 4,
    "delay": 25,
    "percentage": 1
    }

ema_test, sp_test, ap_test, givenf0_test, scaler_f0, scaler_sp, \
        scaler_ap = preprocessing.load_test(
            options["delay"],
            options["percentage"])

# From checkpoints load the keras models
mfcc_model = load_model("checkpoints/model_sp_new.hdf5")
bap_model = load_model("checkpoints/model_bap_new.hdf5")



mfcc_unnormalised = mfcc_model.predict(ema_test).astype(np.float64)
bap_unnormalised = bap_model.predict(ema_test).astype(np.float64)
f0 = givenf0_test.astype(np.float64)
fs = 16000

mfcc_normalised = np.copy(mfcc_unnormalised)
bap_normalised = np.copy(bap_unnormalised)

print(mfcc_normalised.shape)
for i in range(len(scaler_sp)):
    mfcc_normalised[:,:,i] = scaler_sp[i].inverse_transform(mfcc_unnormalised[:,:,i])

for i in range(len(scaler_ap)):
    bap_normalised[:,:,i] = scaler_ap[i].inverse_transform(bap_unnormalised[:,:,i])
    
#plt.plot(f0[0,:])
#plt.show()
#plt.plot(mfcc_normalised[0,:,:])
#plt.show()
#plt.plot(bap_normalised[0,:,:])
#plt.show()

preprocessing.debug_resynth(f0[options["id"],:],
                            mfcc_normalised[options["id"],:],
                            bap_normalised[options["id"],:],fs)
