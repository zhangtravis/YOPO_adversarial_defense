'''WResNet-32 model.'''

import keras.layers
import keras.backend as K
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras.models import Model

K.set_learning_phase(1)

def basic_block(inputs, planes, stride=1, downsample=None):
    x = Conv2D(planes, 3, strides=stride, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4), use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(planes, 3, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization()(x)

    if downsample:
        inputs = downsample(inputs)
        inputs = BatchNormalization()(inputs)
    x = keras.layers.add([x, inputs])
    x = Activation('relu')(x)
    return x


def resnet(inputs, layers, width=1, num_classes=10):
    # WResNet-32 has width = 10
    #inputs = Input(shape=input_shape)

    inplanes = 16
    x = Conv2D(inplanes, 3, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    layer1_out = x

    nblock = [16 * width, 32 * width, 64 * width]
    nstride = [1, 2, 2]

    for l in range(3):
        if l > 0 or inplanes != nblock[l]:
            x = basic_block(x, nblock[l], nstride[l],
                            Conv2D(nblock[l], 1, strides=nstride[l],
                                   use_bias=False, padding='same',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=l2(1e-4)))
        else:
            x = basic_block(x, nblock[l], nstride[l])

        inplanes = nblock[l]
        for _ in range(1, layers[l]):
            x = basic_block(x, inplanes)

    x = AveragePooling2D(pool_size=7)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    #model = Model(inputs=inputs, outputs=outputs)
    #return model, layer1_out
    return outputs, layer1_out


#def resnet_32(input_shape, **kwargs):
#    return resnet(input_shape, [5, 5, 5], **kwargs)