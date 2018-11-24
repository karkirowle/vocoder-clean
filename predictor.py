

from keras.models import load_model
import preprocessing

ema_train, ema_test,f0_train, \
f0_test, sp_train, sp_test, \
ap_train, ap_test, givenf0_train, \
givenf0_test, wavdata, \
scaler_f0, scaler_sp, scaler_ap = preprocessing.load_data(
    options["delay"],
    options["percentage"])

# From checkpoints load the keras models
mfcc_model = load_model("checkpoints/model_sp_new.hdf5")
bap_model = load_model("checkpoints/model_ap_new.hdf5")


options = {
    "id": 1
    }

mfcc = mfcc_model.predict(ema_test[options["id"],:])
bap = bap_model.predict(ema_test[options["id"],:])
f0 = f0_test[options["id"],:])
fs = 16000

preprocessing.debug_resynth(f0,mfcc,bap,fs)
