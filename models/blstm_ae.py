from keras.models import Model

from keras.layers import Input, GaussianNoise, Bidirectional, CuDNNLSTM, TimeDistributed, Dense,Concatenate

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

        # Bottleneck encoder
        dense_ema = TimeDistributed(Dense(options["out_features"]))(gru_4a)
        dense_latent = TimeDistributed(Dense(options["latent_features"]))(gru_4a)

        dense_concat = Concatenate()([dense_ema,dense_latent])

        # Bottleneck decoder
        ema_input = Input(shape=(None,
                                 options["latent_features"] +
                                 options["out_features"]))
        
        latent_input = Input(shape=(None,
                                    options["latent_features"]))
        noise_latent = GaussianNoise(0.01)(latent_input)
        decode_input = Concatenate()([ema_input,noise_latent])
     
        # Decoder
        gru_1a_d = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))
        gru_2a_d = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))
        gru_3a_d = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))
        gru_4a_d = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))

        
        dense = TimeDistributed(Dense(options["num_features"]))

        dense_train = dense(gru_4a_d(gru_3a_d(gru_2a_d(gru_1a_d(dense_concat)))))
        dense_decode = dense(gru_4a_d(gru_3a_d(gru_2a_d(gru_1a_d(ema_input)))))

        self.encode_trainer = Model(inputs,dense_ema)
        self.decode_trainer = Model([ema_input,latent_input],dense_decode)
        self.trainer = Model(inputs,[dense_train,dense_ema])

        self.encode = Model(inputs,dense_concat)
        self.decode = Model(ema_input,dense_decode)

        






        
    
