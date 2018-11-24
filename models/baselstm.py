

import datetime
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from keras import optimizers, callbacks
import models.helpers as H 
# RENAME FOR EVERY SINGLE MODEL
NAME = "logs/dense"

def __main__():
    print("something")

def run(options, logged_options, input_data, output_data):
    # Encoder
    ema_inputs = Input(shape=(options["max_input_length"],options["num_features"]))
    dense_1 = Dense(options["dense_1"], activation="relu")(ema_inputs)
    flatten = Flatten()(dense_1)
    dense_2 = Dense(options["dense_2"])(flatten)
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
              validation_split = 0.2,
              epochs=options["epochs"],
              callbacks=[tb])
    
