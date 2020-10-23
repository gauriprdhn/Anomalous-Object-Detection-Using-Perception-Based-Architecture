from tensorflow.keras.layers import BatchNormalization,Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.layers import Reshape, Input, Activation, Flatten,Dense
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth, filters=(16, 32), latentDim=32):
        inputShape = (width, height, depth)
        chanDim = -1
        # define the input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs
        # loop over the number of filters
        for f in filters:
            # apply a CONV => RELU => BN operation
            initializer = GlorotNormal()
            x = Conv2D(f, (3, 3), strides=2,kernel_initializer = initializer, padding="same")(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = BatchNormalization(axis=chanDim)(x)
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)
        encoder = Model(inputs, latent, name="encoder")
        print(encoder.summary())
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        for f in filters[::-1]:
            initializer = GlorotNormal()
            x = Conv2DTranspose(f, (3, 3), strides=2, kernel_initializer = initializer, padding="same")(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = BatchNormalization(axis=chanDim)(x)
        x = Conv2DTranspose(depth, (3, 3),padding="same")(x)
        outputs = Activation("sigmoid")(x)
        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")
        print(decoder.summary())
        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),
            name="autoencoder")
        print(autoencoder.summary())
        return (encoder, decoder, autoencoder)