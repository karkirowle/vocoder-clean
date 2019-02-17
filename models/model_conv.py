from keras.models import Model

from keras.layers import Input, GaussianNoise, Dense, BatchNormalization
from keras.layers import Conv2D, Reshape
class LSTM_Model(object):
    
    def __init__(self,options):

        inputs = Input(shape=(1000,options["num_features"]))
        noise = GaussianNoise(options["noise"])(inputs)
        reshape = Reshape(target_shape=(1000,options["num_features"],1))(noise)
        conv1 = BatchNormalization()(Conv2D(45,(3,3),padding="same")(reshape))
        conv2 = BatchNormalization()(Conv2D(45,(3,3),padding="same")(conv1))
        conv3 = BatchNormalization()(Conv2D(45,(3,3),padding="same")(conv2))
        conv4 = BatchNormalization()(Conv2D(45,(3,3),padding="same")(conv3))
        conv5 = BatchNormalization()(Conv2D(1,(3,3),padding="same")(conv4))
        reshape2 = Reshape(target_shape=(1000,options["num_features"]))(conv5)
        dense = Dense(options["out_features"])(reshape2)
        self.trainer = Model(inputs, dense)
        
    
