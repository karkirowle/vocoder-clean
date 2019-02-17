from keras.models import Model

from keras.layers import Input, GaussianNoise, Bidirectional, CuDNNLSTM, TimeDistributed, Dense, Reshape
from keras.layers import Conv2D
from keras.applications.inception_v3 import InceptionV3
class LSTM_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(1000,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)
        reshape = Conv2D(64, (3,3), padding="same")(noise)
        # create the base pre-trained model
        base_model = InceptionV3(input_tensor=reshape,
                                 weights='imagenet',
                                 include_top=False)

        for layer in base_model.layers:
            layer.trainable = False

        # LSTM layers share number of hidden layer parameter
        gru_1a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(base_model)
        gru_2a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_1a)
        gru_3a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_2a)

        # Densex
        dense = Dense(options["out_features"])(gru_3a)
        self.trainer = Model(inputs, dense)
        
    
