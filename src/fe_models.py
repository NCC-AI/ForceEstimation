
from keras.layers import *
from keras.models import Model
from keras.applications import *

class ModelMaker():

    def __init__(self, input_shape, backbone):
        self.input_shape = input_shape
        self.backbone    = backbone
        self.cfg         = {'VGG16': ['F', 512, 256, 'D'],
                           }

    def make_top(self, net):
        extractor   = self.extractor()
        for x in self.cfg[self.backbone]:
            if x == 'F':
                net = Flatten(input_shape=extractor.output_shape[1:])(net)
            elif x == 'D':
                net = Dropout(0.5)(net)
            else:
                net = Dense(x, activation='relu')(net)
        return net

    def make_last(self, net):
        net = Dense(1)(net)
        return net

    def extractor(self):
        model = eval(self.backbone)(include_top=False, weights='imagenet', input_tensor=Input(shape=self.input_shape))
        return model

    def regressor(self, weights=None):
        extractor       = self.extractor()
        input_tensor    = Input(shape=extractor.output_shape[1:])
        output_tensor   = self.make_last(self.make_top(input_tensor))
        model           = Model(inputs=input_tensor, outputs=output_tensor)
        if not weights == None:
            model.load_weights(weights)
        return model

    def predictor(self, weights=None):
        extractor       = self.extractor()
        input_tensor    = extractor.input
        output_tensor   = self.make_last(self.make_top(extractor.output))
        model           = Model(inputs=input_tensor, outputs=output_tensor)
        if not weights == None:
            model.load_weights(weights, by_name=True)
        return model


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

def dense_layers(cfg, net):
    for x in cfg:
        if x == 'M':
            net = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(net)
        else:
            net = Conv2D(x, (3, 3), padding='same')(net)
            net = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(net)
            net = Activation('relu')(net)
    return net

def lstm_layers(cfg, net):
    for x in cfg:
        if x == 'M':
            net = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(net)
        else:
            net = TimeDistributed(Conv2D(x, (3, 3), activation='relu', padding='same'))(net)
            #net = TimeDistributed(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))(net)
            #net = TimeDistributed(Activation('relu'))(net)
    return net

def CONVLSTM2D(input_shape, backbone='VGG16'):

    input_tensor = Input(shape=input_shape)
    net          = input_tensor
    net          = lstm_layers(cfg[backbone], net)
    net          = Flatten()(net)
    net          = Dense(512, activation='relu')(net)
    net          = Dense(128, activation='relu')(net)
    predictions  = Dense(1)(net)

    model = Model(input_tensor, predictions)
    return model

def DENSELSTM(input_shape, backbone='VGG16'):

    input_tensor = Input(shape=input_shape)
    net          = input_tensor
    net          = lstm_layers(cfg[backbone], net)
    net          = TimeDistributed(Flatten())(net)
    net          = LSTM(256, return_sequences=False)(net)
    net          = Dense(128, activation='relu')(net)
    predictions  = Dense(1)(net)

    model = Model(input_tensor, predictions)
    return model

def TRANSFER(input_shape, backbone='VGG16'):

    base_model   = eval(backbone)(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))
    net          = Flatten()(base_model.output)
    net          = Dense(512, activation='relu')(net)
    net          = Dense(256, activation='relu')(net)
    predictions  = Dense(1)(net)

    model = Model(base_model.input, predictions)
    for layer in bese_model.layers:
        layer.trainable = False
    return model