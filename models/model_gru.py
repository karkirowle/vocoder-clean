from keras.models import Model

from keras.layers import Input, GaussianNoise, CuDNNGRU, TimeDistributed, Dense

class GRU_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(None,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)

        # GRU layers share number of hidden layer parameter
        gru_1a = CuDNNGRU(options["gru"], return_sequences=True)(noise)
        gru_2a = CuDNNGRU(options["gru"], return_sequences=True)(gru_1a)
        gru_3a = CuDNNGRU(options["gru"], return_sequences=True)(gru_2a)
        gru_4a = CuDNNGRU(options["gru"], return_sequences=True)(gru_3a)

        # Dense
        dense = TimeDistributed(Dense(options["out_features"]))(gru_4a)
        self.trainer = Model(inputs, dense)
        
    
