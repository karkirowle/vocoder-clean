
from experiment import ex
import numpy as np

# Cross-validation file

options = {
    "experiment" : "reproduction_save_attempt",
    "max_input_length" : 1000,
    "num_features": 15,
    "lr": 0.01, # 0.003 # not assigned in Takuragi paper
    "clip": 5,
    "epochs": 60, #60
    "out_features": 82,
    "gru": 128,
    "seed": 10,
    "noise": 0,
    "delay": 1, # 25 
    "batch_size": 45, #90
    "percentage": 1,
    "k": 0,
    "total_samples": 2274,
    "save_dir": "processed_comb2_filtered_2"
}


mcd = np.zeros((10))

for i in range(10):
    options["k"] = i
    ex.add_config(options)
    r = ex.run()
    mcd[i] = r.result
    np.save("mcd.npy", mcd)

print("10-fold CV: ", np.mean(mcd))

