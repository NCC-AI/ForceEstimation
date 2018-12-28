from keras.layers import *
from keras.models import Model
from keras.applications import *


def make_top_layers(net):
    """ Get layer and Return layer, not model.
    """
    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(256, activation='relu')(net)
    net = Dropout(0.5)(net)
    net = Dense(1)(net)
    
    return net


class MomentModel():

    def __init__(self, input_shape=(299, 299, 3), backbone='VGG16'):
        self.input_shape = input_shape
        self.backbone = backbone
        self.base_model = eval(backbone)(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))
        self.cfg = {'VGG16': ['F', 512, 256, 'D'],
        }

    def top_layers(self, net):
        for x in self.cfg[self.backbone]:
            if x == 'F':
                net = Flatten(input_shape=self.base_model.output_shape[1:])(net)
            elif x == 'D':
                net = Dropout(0.5)(net)
            else:
                net = Dense(x, activation='relu')(net)
        net = Dense(1)(net)
        return net

    def top_model(self, weights=None):
        input_tensor = Input(shape=self.base_model.output_shape[1:])
        prediction = self.top_layers(input_tensor)
        model = Model(inputs=input_tensor, outputs=prediction)
        if not weights == None:
            model.load_weights(weights)
        return model

    def model(self, weights=None):
        prediction = self.top_layers(self.base_model.output)
        model = Model(inputs=self.base_model.input, outputs=prediction)
        if not weights == None:
            model.load_weights(weights, by_name=True)
        return model


class SeriesModel():

    def __init__(self, input_shape=(5, 299, 299, 3), backbone='VGG16'):
        self.input_shape = input_shape
        self.backbone = backbone
        self.cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

    def base_layers(self, net):
        for x in self.cfg[self.backbone]:
            if x == 'M':
                net = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(net)
            else:
                net = TimeDistributed(Conv2D(x, (3, 3), padding='same'))(net)
                net = TimeDistributed(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))(net)
                net = TimeDistributed(Activation('relu'))(net)
        return net

    def top_layers(self, net):
        net = TimeDistributed(Flatten())(net)
        net = LSTM(1024, return_sequences=True)(net)
        net = Dense(256, activation='relu')(net)
        net = Dense(1)(net)
        return net

    def model(self, weights=None):
        input_tensor = Input(shape=self.input_shape)
        prediction = self.top_layers(self.base_layers(input_tensor))
        model = Model(inputs=input_tensor, outputs=prediction)
        if not weights == None:
            model.load_weights(weights, by_name=True)
        return model