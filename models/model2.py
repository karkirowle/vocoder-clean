

import datetime
from keras.layers import Flatten, GaussianNoise, Dense, Input, Conv1D, Dropout
from keras.models import Model
from keras import optimizers, callbacks
import models.helpers as H 
# RENAME FOR EVERY SINGLE MODEL
NAME = "logs/dense2_noise"

def __main__():
    print("something")

def run(options, logged_options, input_data, output_data):
    # Encoder
    ema_inputs = Input(shape=(options["max_input_length"],options["num_features"]))
    noise = GaussianNoise(options["noise"])(ema_inputs)
    conv_1 = Conv1D(16, 100, padding="same", activation="relu")(noise)
    conv_2 = Conv1D(8, 100, padding="same", activation="relu")(conv_1)
    conv_3 = Conv1D(1, 100, padding="same", activation="relu")(conv_2)
    flatten = Flatten()(conv_3)
    dense_1 = Dropout(options["droprate"])(Dense(options["dense_2"], activation="relu")(flatten))
    dense_2 = Dense(options["dense_3"])(dense_1)

    model = Model(ema_inputs, dense_2)
    print(model.summary())
    
    optimiser = H.optimiser_handler(options)
    model.compile(loss="mse", optimizer=optimiser)

    # Formulate path name for recording
    path_name = H.path_name_inf(options,logged_options,NAME)

    # Record the experiment performed
    H.experiment_logger(options)
    tb = callbacks.TensorBoard(log_dir=path_name)

    
    model.fit(input_data, output_data,
              shuffle="True",
              validation_split = 0.1,
              epochs=options["epochs"],
              callbacks=[tb])

    return model
