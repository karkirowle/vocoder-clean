

import datetime
from keras.layers import BatchNormalization, Flatten, LSTM, TimeDistributed, Dense, Input, Conv1D, CuDNNLSTM, GaussianNoise
from keras.models import Model
from keras import optimizers, callbacks
import models.helpers as H 
# RENAME FOR EVERY SINGLE MODEL
NAME = "logs/shallow_lstm_preproc"

def __main__():
    print("something")

def run(options, logged_options, input_data, output_data):

    # Model-only output reshaping
    output_data = output_data.reshape(output_data.shape[0], output_data.shape[1], 1)
    # Encoder
    ema_inputs = Input(shape=(None,options["num_features"]))
    noise = GaussianNoise(options["noise"])(ema_inputs)
    LSTM_layer_1 = CuDNNLSTM(options["gru"], return_sequences=True)(noise)
    LSTM_layer_2 = CuDNNLSTM(options["gru"], return_sequences=True)(LSTM_layer_1)
    Dense_1 = TimeDistributed(Dense(1024, activation="relu"))(LSTM_layer_2)
    Dense_2 = BatchNormalization()(TimeDistributed(Dense(1024, activation="relu"))(Dense_1))
    Dense_3 = TimeDistributed(Dense(1))(Dense_2)
    model = Model(ema_inputs, Dense_3)
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
              batch_size=1,
              shuffle="True",
              validation_split = 0.2,
              epochs=options["epochs"],
              callbacks=[tb])
    
