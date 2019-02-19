from keras.models import Model

from keras.layers import Input, GaussianNoise, Bidirectional, CuDNNLSTM, TimeDistributed, Dense
from keras.layers import Conv2D, BatchNormalization, Reshape
class LSTM_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(None,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)

        reshape = Reshape(target_shape=(1000,options["num_features"],1))(noise)
        conv1 = BatchNormalization()(Conv2D(16,(5,5),padding="same",activation="relu")(reshape))
        conv2 = BatchNormalization()(Conv2D(16,(5,5),padding="same",activation="relu")(conv1))
        reshape2 = Reshape(target_shape=(1000,-1))(conv2)

        # LSTM layers share number of hidden layer parameter
        gru_1a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(reshape2)
        gru_2a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_1a)
        gru_3a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_2a)
        gru_4a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_3a)

        # Densex
        dense = Dense(options["out_features"])(gru_4a)
        self.trainer = Model(inputs, dense)
        
    
