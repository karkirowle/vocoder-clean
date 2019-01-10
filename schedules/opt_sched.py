
import numpy as np
import math
from keras import optimizers
from keras.callbacks import LearningRateScheduler
def taguchi_opt():
    rmsprop_optimiser = optimizers.RMSprop(lr=0.01,
                                           clipvalue=5)
    return rmsprop_optimiser

def zhengcheng_opt():

    def exp_decay(epoch):
                  initial_lrate = 0.01
                  drop = 0.5
                  epochs_drop = 13
                  lrate = initial_lrate * math.pow(drop,
                                                   math.floor((1+epoch)/
                                                              epochs_drop))
                  return lrate

    lrate = LearningRateScheduler(exp_decay)
    optimiser = optimizers.SGD(lr=0.01)

    return optimiser, lrate
