

import datetime
from keras.layers import Flatten, TimeDistributed, Dense, Input, Conv1D, CuDNNLSTM, GaussianNoise
from keras.models import Model
from keras import optimizers, callbacks
import models.helpers as H 
# RENAME FOR EVERY SINGLE MODEL
NAME = "logs/lstm_preproc"

def __main__():
    print("something")

def run(options, logged_options, input_data, output_data):

    # Model-only output reshaping
    output_data = output_data.reshape(output_data.shape[0], output_data.shape[1], 1)
    # Encoder
    ema_inputs = Input(shape=(None,options["num_features"]))
    noise = GaussianNoise(options["noise"])(ema_inputs)
    Dense_P = TimeDistributed(Dense(options["dense_1"], activation="tanh"))(noise)
    LSTM_layer_1 = CuDNNLSTM(options["gru"], return_sequences=True)(Dense_P)
    LSTM_layer_2 = CuDNNLSTM(options["gru"], return_sequences=True)(LSTM_layer_1)
    LSTM_layer_3 = CuDNNLSTM(options["gru"], return_sequences=True)(LSTM_layer_2)
    LSTM_layer_4 = CuDNNLSTM(options["gru"], return_sequences=True)(LSTM_layer_3)
    Dense_1 = TimeDistributed(Dense(1))(LSTM_layer_4)
    model = Model(ema_inputs, Dense_1)
    print(model.summary())
    
    optimiser = H.optimiser_handler(options)
    model.compile(loss="mse", optimizer=optimiser)

    # Formulate path name for recording
    path_name = H.path_name_inf(options,logged_options,NAME)

    # Record the experiment performed
    H.experiment_logger(options)
    tb = callbacks.TensorBoard(log_dir=path_name)
    es = callbacks.EarlyStopping(monitor="val_loss", mode="min", min_delta=1, patience=10)
    
    model.fit(input_data, output_data,
              shuffle="True",
              validation_split = 0.2,
              epochs=options["epochs"],
              batch_size = 50,
              callbacks=[tb,es])
    