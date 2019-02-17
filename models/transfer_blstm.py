from keras.models import Model

from keras.layers import Input, GaussianNoise, Bidirectional, CuDNNLSTM, TimeDistributed, Dense, Reshape
from keras.layers import Conv2D
from keras.backend import repeat_elements
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
class LSTM_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(1000,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)
        reshape = Reshape(target_shape=(1000,options["num_features"],1))(noise)
        print("azis")
        repeater1 = repeat_elements(reshape, 6, 2)
        repeater2 = repeat_elements(repeater1, 3, 3)
        print(repeater2.shape)
        print("ezis")
        # create the base pre-trained model
        base_model = InceptionV3(input_tensor=repeater2,
                                 weights='imagenet',
                                 include_top=False)
        print("ez")
        for layer in base_model.layers:
            layer.trainable = False

        print(base_model.output.shape)
        # LSTM layers share number of hidden layer parameter
        gru_1a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(base_model.output)
        print("az")
        gru_2a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_1a)
        gru_3a = Bidirectional(CuDNNLSTM(options["gru"],
                                        return_sequences=True))(gru_2a)

        # Densex
        dense = Dense(options["out_features"])(gru_3a)
        self.trainer = Model(inputs, dense)
        
    
