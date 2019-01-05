
import sys
sys.path.insert(0,'../..')
import data_loader
from scipy.signal import correlate
import matplotlib.pyplot as plt
import numpy as np
# Script to determine the delay

options = {}
options["batch_size"] = 2200
options["delay"] = 0
options["k"] = 0
options["num_features"] = 15
options["save_dir"] = "../../processed_comb2_filtered_3"
options["out_features"] = 82
train_gen = data_loader.DataGenerator(options,True,True)
X, Y = train_gen.__getitem__(0)

ac = np.zeros((options["batch_size"],1999))
lag = np.linspace(-999,999,1999)
print(lag.shape)
for i in range(options["batch_size"]):
    X_id = X[i,:,1]
    Y_id = Y[i,:,1]

    ac[i,:] = correlate(X_id,Y_id)
   
ac_mean = np.mean(ac,axis=0)

plt.plot(lag,ac_mean)
plt.show()



