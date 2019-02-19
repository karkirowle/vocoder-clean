from keras.models import Model

from keras.layers import Input, GaussianNoise, Bidirectional, CuDNNLSTM, TimeDistributed, Dense
from keras.layers import Activation

from keras_layer_normalization import LayerNormalization
#from preprocessing3 import mlpg_postprocessing
from keras.layers import Layer, Dropout, Lambda
from sklearn.externals import joblib
import sys
sys.path.insert(0,'..')
import preprocessing3 as proc

#class MLPGLayer(Layer):
#    def __init__(self, output_dim, options, **kwargs):
#        print("init")
#        self.output_dim = output_dim
#        self.options = options
#        super(MLPGLayer, self).__init__(**kwargs)
#
#    def build(self, input_shape):
#        print("build")
#        super(MLPGLayer, self).build(input_shape)
#        
#    def call(self, inputs):
#        save_dir = self.options["save_dir"]
#        scaler_sp = joblib.load(save_dir + '/scaler_sp_.pkl')
#        print(inputs)
#        
#        return proc.mlpg_postprocessing(inputs,self.options,scaler_sp)
#
#    def compute_output_shape(self, input_shape):
#        print("cos")
#        return (input_shape[0], input_shape[1], self.output_dim)

class GRU_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(None,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)

        dense1 = Dropout(0.5)(Dense(128,
                                    kernel_initializer="lecun_normal")(noise))
        layernorm1 = Activation("sigmoid")(LayerNormalization()(dense1))
        
        dense2 = Dropout(0.5)(Dense(128,
                                    kernel_initializer="lecun_normal")(layernorm1))
        layernorm2 = Activation("sigmoid")(LayerNormalization()(dense2))

        dense3 = Dropout(0.5)(Dense(128,
                                    kernel_initializer="lecun_normal")(layernorm2))
        layernorm3 = Activation("sigmoid")(LayerNormalization()(dense3))

        gru_1a = Dropout(0.5)(Bidirectional(CuDNNLSTM(256,
                                         kernel_initializer="lecun_normal",
                                        return_sequences=True))(layernorm3))
        gru_2a = Dropout(0.5)(Bidirectional(CuDNNLSTM(256,
                                         kernel_initializer="lecun_normal",
                                        return_sequences=True))(gru_1a))
        # Dense
        dense = Dense(options["out_features"],
                      kernel_initializer="lecun_normal")(gru_2a)

        self.trainer = Model(inputs,dense)
    

