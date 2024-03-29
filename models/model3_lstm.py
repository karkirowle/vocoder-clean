

import datetime
from keras.layers import BatchNormalization,Flatten, TimeDistributed, Dense, Input, Conv1D, CuDNNLSTM, GaussianNoise
from keras.models import Model
from keras import optimizers, callbacks
import models.helpers as H 
# RENAME FOR EVERY SINGLE MODEL
NAME = "logs/lstm_preproc"

def __main__():
    print("something")

def run(options, logged_options, input_data, output_data, input_test, output_test):

    # Model-only output reshaping
    output_data = output_data.reshape(output_data.shape[0], output_data.shape[1], options["out_features"])
    output_test = output_test.reshape(output_test.shape[0], output_test.shape[1], options["out_features"])
    # Encoder
    ema_inputs = Input(shape=(None,options["num_features"]))
    noise = GaussianNoise(options["noise"])(ema_inputs)
    #Dense_P = TimeDistributed(Dense(options["dense_1"], activation="tanh"))(noise)
    GRU_layer_1 = CuDNNLSTM(options["gru"], return_sequences=True)(noise)
    GRU_layer_2 = CuDNNLSTM(options["gru"], return_sequences=True)(GRU_layer_1)
    GRU_layer_3 = CuDNNLSTM(options["gru"], return_sequences=True)(GRU_layer_2)
    GRU_layer_4 = CuDNNLSTM(options["gru"], return_sequences=True)(GRU_layer_3)
    GRU_layer_5 = CuDNNLSTM(options["gru"], return_sequences=True)(GRU_layer_4)
    Dense_1 = TimeDistributed(Dense(options["out_features"]))(GRU_layer_5)
    #Dense_2 = TimeDistributed(Dense(1))(Dense_1)
    model = Model(ema_inputs, Dense_1)
    print(model.summary())
    optimiser = H.optimiser_handler(options)
    model.compile(loss="mse", optimizer=optimiser)

    # Formulate path name for recording
    path_name = H.path_name_inf(options,logged_options,NAME)

    # Record the experiment performed
    H.experiment_logger(options)
    tb = callbacks.TensorBoard(log_dir=path_name)
    es = callbacks.EarlyStopping(monitor="val_loss", mode="min", min_delta=0.001, patience=10, restore_best_weights=True)
    model.fit(input_data, output_data,
              shuffle="True",
              validation_data = (input_test, output_test),
              batch_size = 50,
              epochs=options["epochs"],
              callbacks=[tb,es])

    return model
