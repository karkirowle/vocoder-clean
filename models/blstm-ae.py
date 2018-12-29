from keras.models import Model

from keras.layers import Input, GaussianNoise, Bidirectional, CuDNNLSTM, TimeDistributed, Dense

class LSTM_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(None,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)

        # Encoder
        gru_1a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(noise)
        gru_2a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_1a)
        gru_3a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_2a)
        gru_4a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_3a)

        # Bottleneck
        dense_ema = TimeDistributed(Dense(options["out_features"]))(gru_4a)
        dense_latent = TimeDistributed(Dense(options["latent_features"]))(gru_4a)

        ema_input = Input(shape=(None,
                                 options["latent_features"] +
                                 options["out_features"]))
        # Decoder

        gru_1a_d = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))
        gru_2a_d = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))
        gru_3a_d = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))
        gru_4a_d = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))

        
        dense = TimeDistributed(Dense(options["num_features"]))(gru_4a_d)

        dense_train = dense(gru_4_a_d(gru_3a_d(gru_2a_d(gru_1a_d([dense_ema,
                                                                  dense_latent])))))
        dense_decode = dense(gru_4_a_d(gru_3a_d(gru_2a_d(gru_1a_d(ema_input)))))
        
        self.trainer = Model(inputs,[dense_ema,dense_train])
        self.encode = Model(inputs,[dense_ema,dense_latent])
        self.decode = Model(inputs,ema_input)

        






        
    
