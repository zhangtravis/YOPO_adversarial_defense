import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""  # specify which GPU(s) to be used

import math
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import tensorflow as tf

import keras
import keras.backend as K
from keras.layers import Dense, Conv2D, Input, Flatten, MaxPool2D
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import mnist, cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger #LearningRateScheduler
import pickle
#from keras.utils import multi_gpu_model
from keras.layers import Lambda
from keras.layers import average


def create_Model(x_input_shape):
    inputs = Input(shape = x_input_shape)
    layer1_out = x = Conv2D(filters=32, kernel_size=5)(inputs)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(filters=64, kernel_size=5)(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    softmax = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=softmax)
    return model, layer1_out

# This is the (inputs, targets) generator for training
def adversary_generator(datagen, x, logits):
    old_generator = datagen.flow(x, logits, batch_size=args_batch_size)
    for x_batch, logits_batch in old_generator:
        x_new_batch = np.copy(x_batch)
        for i in range(args_step_num):
            grad = sess.run(grad_t, feed_dict={input_xs: x_new_batch, targets_ys: logits_batch,
                                               sample_weights_ys: [1] * len(x_batch)})
            grad = np.sign(grad)
            x_new_batch += args_step_size * grad
            x_new_batch = np.clip(x_new_batch, x_batch - args_eps, x_batch + args_eps)
            x_new_batch = np.clip(x_new_batch, 0.0, 1.0)
        yield x_new_batch, logits_batch

def yopo_adversary_generator(datagen, x, logits, m, n):
    old_generator = datagen.flow(x, logits, batch_size=args_batch_size)
    for x_batch, logits_batch in old_generator:
        eta = np.random.uniform(-args_eps, args_eps, x_batch.shape)
        x_new_batch = x_batch + eta
        for i in range(m + 1):
            yield x_new_batch, logits_batch
            if i == m:
                break
            # Add perturbation to inputs. loss_layer1 is only computed once.
            loss_layer1 = sess.run(loss_layer1_t, feed_dict={input_xs: x_new_batch, targets_ys: logits_batch,
                                                             sample_weights_ys: [1] * len(x_batch)})
            for j in range(n):
                grad = sess.run(yopo_grad_t, feed_dict={input_xs: x_new_batch, p_layer1_t: loss_layer1})
                grad = np.sign(grad)
                x_new_batch += args_step_size * grad
                x_new_batch = np.clip(x_new_batch, x_batch - args_eps, x_batch + args_eps)
                x_new_batch = np.clip(x_new_batch, 0.0, 1.0)

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # For reproduction
    np.random.seed(2019)
    tf.set_random_seed(9102)

    # Prepare data
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    logits_train = keras.utils.to_categorical(y_train, num_classes)
    logits_test = keras.utils.to_categorical(y_test, num_classes)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    train_datagen = ImageDataGenerator()
    train_datagen.fit(x_train)
    test_datagen = ImageDataGenerator()

    # Prepare model
    model, layer1_out = create_Model(x_input_shape=x_train[0].shape)
    opt = Adam(lr=1e-3, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Set parameters here
    args_step_size = 2.0 / 255.0
    args_eps = 8.0 / 255.0
    args_step_num = 7
    args_batch_size = 64
    args_free_m = 8  # m in Free-m
    args_yopo_m = 3  # m in YOPO-m-n
    args_yopo_n = 5  # n in YOPO-m-n
    args_lr_m = 3  # 1 if standard adversarial training, or use free_m or yopo_m

    input_xs = model.input
    output_ys = model.output
    targets_ys = model.targets[0]
    sample_weights_ys = model.sample_weights[0]
    loss_t = model.total_loss
    grad_t = K.gradients(loss_t, input_xs)[0]  # gradient of loss w.r.t. inputs

    loss_layer1_t = K.gradients(loss_t, layer1_out)[0]  # gradient of loss w.r.t. the output of the first layer
    p_layer1_t = K.placeholder(layer1_out.shape, dtype=tf.float32)
    hamilton_layer1_t = keras.layers.dot([Flatten()(p_layer1_t), Flatten()(layer1_out)], axes=1)
    yopo_grad_t = K.gradients(hamilton_layer1_t, input_xs)[0]  # YOPO approximation of grad_t

    # An example of YOPO-m-n
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    epochs = math.ceil(300 / args_yopo_m)
    model.fit_generator(yopo_adversary_generator(train_datagen, x_train, logits_train, args_yopo_m, args_yopo_n),
                        # Replace this line to use other algorithms
                        validation_data=adversary_generator(test_datagen, x_test, logits_test),
                        validation_steps=math.ceil(len(x_test) / args_batch_size),
                        epochs=epochs, verbose=2, workers=0,
                        steps_per_epoch=math.ceil(len(x_train) / args_batch_size) * (args_yopo_m + 1))
