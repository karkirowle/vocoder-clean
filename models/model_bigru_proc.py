from keras.models import Model

from keras.layers import Input, GaussianNoise, Bidirectional, CuDNNGRU, TimeDistributed, Dense

class GRU_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(None,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)

        
        dense_1 = Dense(128,activation="sigmoid")(noise)
        dense_2 = Dense(128,activation="sigmoid")(dense_1)
        # Because the iput range might not be constrained to sigmoid
        dense_3 = Dense(options["num_features"],activation="linear")(dense_2)

        
        # GRU layers share number of hidden layer parameter
        gru_1a = Bidirectional(CuDNNGRU(150,
                                        return_sequences=True))
        gru_2a = Bidirectional(CuDNNGRU(150,
                                        return_sequences=True))
        gru_3a = Bidirectional(CuDNNGRU(150,
                                        return_sequences=True))
        gru_4a = Bidirectional(CuDNNGRU(150,
                                        return_sequences=True))

        dense = Dense(options["out_features"])
        train_out = dense(gru_4a(gru_3a(gru_2a(gru_1a(noise)))))
        transfer_out = dense(gru_4a(gru_3a(gru_2a(gru_1a(dense_3)))))

        # Dense
        self.trainer = Model(inputs, train_out)
        self.transfer = Model(inputs, transfer_out)
        
    
