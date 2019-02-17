from keras.models import Model

from keras.layers import Input, GaussianNoise, Dense, BatchNormalization
from keras.layers import Conv1D, Reshape
class LSTM_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(1000,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)
        reshape = Reshape(target_shape=(1000,options["num_features"],1))(noise)
        conv1 = BatchNormalization()(Conv1D(45,3,padding="same")(noise))
        conv2 = BatchNormalization()(Conv1D(45,3,padding="same")(conv1))
        conv3 = BatchNormalization()(Conv1D(45,3,padding="same")(conv2))
        conv4 = BatchNormalization()(Conv1D(45,3,padding="same")(conv3))
        conv5 = BatchNormalization()(Conv1D(45,3,padding="same")(conv4))

        # Dense
        dense = Dense(options["out_features"])(conv5)
        self.trainer = Model(inputs, dense)
        
    
