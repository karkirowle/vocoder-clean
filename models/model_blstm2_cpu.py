from keras.models import Model

from keras.layers import Input, GaussianNoise, Bidirectional, LSTM, TimeDistributed, Dense

class LSTM_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(None,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)

        # LSTM layers share number of hidden layer parameter
        gru_1a = Bidirectional(LSTM(options["gru"],
                                        return_sequences=True))(noise)
        gru_2a = Bidirectional(LSTM(options["gru"],
                                        return_sequences=True))(gru_1a)
        gru_3a = Bidirectional(LSTM(options["gru"],
                                        return_sequences=True))(gru_2a)
        gru_4a = Bidirectional(LSTM(options["gru"],
                                        return_sequences=True))(gru_3a)

        # Densex
        dense = Dense(options["out_features"])(gru_4a)
        self.trainer = Model(inputs, dense)
        
    
