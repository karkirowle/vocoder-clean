from keras.models import Model

from keras.layers import Input, GaussianNoise, Bidirectional, CuDNNLSTM, TimeDistributed, Dense
from keras_layer_normalization import LayerNormalization
class GRU_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(None,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)

        dense1 = Dense(128,activation="tanh")(noise)
        dense2 = Dense(128,activation="tanh")(dense1)
        dense3 = Dense(128,activation="tanh")(dense2)
        L_Norm = LayerNormalization()(dense3)
        # GRU layers share number of hidden layer parameter
        gru_1a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(L_Norm)
        gru_2a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_1a)
        # Dense
        dense = Dense(options["out_features"])(gru_2a)
        self.trainer = Model(inputs, dense)
        
    
