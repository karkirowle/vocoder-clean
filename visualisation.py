from keract import get_activations, get_gradients_of_activations
import data_loader

from keras.models import load_model
import matplotlib.pyplot as plt
import audio
import numpy as np
import preprocessing3 as proc
import sounddevice as sd
# Load validation data


def visualise(args):
    options = {
        "bins_1": 41,
        "seed": 25, #10
        "delay": 0, # 25
        "batch_size": 30,
        "percentage": 1,
        "k": 0,
        "save_dir": "processed_comb_test_3_padded",
    }

    # Load the model target for visualisations
    model = load_model("checkpoints/" + args.model +
                       "0" +
                       ".hdf5")

    # Load the dataset, but first get validation size
    N = proc.fetch_validation_size(options,args.dataset,model)
    options["batch_size"] = N
    val_gen = data_loader.DataGenerator(options,
                                        False,
                                        False,
                                        False,
                                        args.shift,
                                        label=args.dataset)

    # Perform feedforward pass to obtain MFCCS
    mfcc_normalised = model.predict_generator(val_gen)



    ema_test, sp_test, _ = val_gen.__getitem__(0)

    f0 = data_loader.load_puref0(options["save_dir"],
                                 options["k"],args.dataset).astype(np.float64)
    bap_gt_u = data_loader.load_bap(options["save_dir"],
                                    options["k"],args.dataset)
    scaler_sp = data_loader.load_scalersp(options["save_dir"])

    mlpg_generated = proc.mlpg_postprocessing(mfcc_normalised,
                                          options["bins_1"],
                                          scaler_sp)

    id = [1,10] 
    act = get_activations(model,ema_test[id,:,:])
    print(act)
    #plt.plot(f0[id,:])
    #plt.show()
    #resynth_length = len(np.trim_zeros(f0[id,:],'b'))
    
    #print(resynth_length)

    #sound1 = audio.debug_resynth(f0[id,:resynth_length],
    #                        mlpg_generated[id,:resynth_length,:],
    #                        bap_gt_u[id,:resynth_length,:],
    #                        fs=16000,
    #                        an=5)

    input = act.get('input_1:0')
    blstm1 = act.get('bidirectional_1/concat_2:0')
    blstm2 = act.get('bidirectional_2/concat_2:0')
    blstm3 = act.get('bidirectional_3/concat_2:0')
    blstm4 = act.get('bidirectional_4/concat_2:0')
    dense1 = act.get('dense_1/add:0')
    blstms = [blstm1, blstm2, blstm3, blstm4]

    c = 0.2
    diff_blstm1 = np.diff(blstm1,axis=1) > c

    c = 0.5
    diff_blstm2 = np.diff(blstm2,axis=1) > c
    c = 0.4
    diff_blstm3 = np.diff(blstm3,axis=1) > c
    c = 0.3
    diff_blstm4 = np.diff(blstm4,axis=1) > c

    diff_blstms = [diff_blstm1, diff_blstm2, diff_blstm3, diff_blstm4]
    
    fig = plt.figure(num=1, figsize=(8.2,8.2), dpi=80, facecolor="w", edgecolor="k")

    for index in range(len(id)):
        plt.subplot(2,len(id),index+1)
        im = plt.imshow(diff_blstm1[index,:,:].T,cmap="hot",aspect="auto")
        it1 = plt.title("Layer 1 Example " + str(index+1))
        it1.set_fontsize(10)
        plt.subplot(2,len(id),len(id)+index+1)
        im = plt.imshow(diff_blstm4[index,:,:].T,cmap="hot",aspect="auto")
        it2 = plt.title("Layer 4 Example " + str(index+1))
        it2.set_fontsize(10)
        if (index == 0):
            it3 = plt.xlabel("time [samples]")
            it3.set_fontsize(8)
            it4 = plt.ylabel("filter activation")
            it4.set_fontsize(8)
    #plt.show()
    
    #for index in range(len(id)):
    #    for i,blstm in enumerate(diff_blstms):
    #        plt.subplot(2,3,i+1)
    #        im = plt.imshow(blstm[index,:,:].T,cmap="hot", aspect="auto")
    #        plt.title("Layer " + str(i))
    #    plt.show()
    plt.savefig("paper/blstm_act.pgf")

    intervals = [0,80,100,140,180]

    import time
    
    for i in range(len(intervals)-1):
        
        sound1 = audio.debug_resynth(f0[id,intervals[i]:intervals[i+1]],
                                mlpg_generated[id,intervals[i]:intervals[i+1],:],
                                bap_gt_u[id,intervals[i]:intervals[i+1],:],
                                fs=16000,
                                an=5)
        time.sleep(5)
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
                                              "d11", "d12"])
    args = parser.parse_args()
    visualise(args)
