from keras.models import Model

from keras.layers import Input, GaussianNoise, Bidirectional, CuDNNLSTM, TimeDistributed, Dense

class LSTM_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(None,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)

        # LSTM layers share number of hidden layer parameter
        gru_1a = Bidirectional(CuDNNLSTM(128,
                                        return_sequences=True))(noise)
        gru_2a = Bidirectional(CuDNNLSTM(128,
                                        return_sequences=True))(gru_1a)
        gru_3a = Bidirectional(CuDNNLSTM(128,
                                        return_sequences=True))(gru_2a)
        gru_4a = Bidirectional(CuDNNLSTM(128,
                                        return_sequences=True))(gru_3a)

        # Densex
        dense = Dense(options["out_features"])(gru_4a)
        self.trainer = Model(inputs, dense)
        
    
