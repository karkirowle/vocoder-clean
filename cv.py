
from experiment import ex
import numpy as np

# Cross-validation file

options = {
    "experiment" : "reproduction_save_attempt",
    "max_input_length" : 1000,
    "num_features": 15,
    "lr": 0.01, # 0.003 # not assigned in Takuragi paper
    "clip": 5,
    "epochs": 100, #60
    "out_features": 82,
    "gru": 128,
    "seed": 10,
    "noise": 0,
    "delay": 0, # 25 
    "batch_size": 90, #90
    "percentage": 1,
    "k": 0,
    "total_samples": 2274,
    "save_dir": "processed_comb2_filtered_3"
}


train = False

if (train):
    mcd = np.zeros((10))

    for i in range(10):
        options["k"] = i
        ex.add_config(options)
        r = ex.run()
        mcd[i] = r.result
        np.save("mcd2.npy", mcd)

    print("10-fold CV: ", np.mean(mcd))
else:
    mcd = np.load("mcd2.npy")
    print("10-fold CV std: ", np.std(mcd))
    
    
