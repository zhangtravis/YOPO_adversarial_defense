import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used

import math
import numpy as np
import tensorflow as tf
import time

import pickle

import keras
import keras.backend as K
from keras.layers import Dense, Conv2D, Input, Flatten, MaxPool2D
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
import pickle
#from keras.utils import multi_gpu_model
from keras.layers import Lambda
from keras.layers import average
from script_cifar.base_resnet_32 import resnet as create_Model_single
import matplotlib.pyplot as plt

def reduce_mean(softmax):
    return tf.reduce_mean(softmax, 0)

def crop(x_input, loc_x, loc_y ):
    return x_input[:, loc_x - 14:loc_x + 14, loc_y - 14:loc_y + 14, :]
    #return x_input

#loc = [[14, 14]]
loc = [[14, 14], [14, 16], [14, 18], [16, 14], [16, 18], [18, 14], [18, 16], [18, 18]]


def create_Model(x_input_shape):
    layer1_out_list = []
    softmax_list = []
    inputs = Input(shape = x_input_shape)
    for i, loc_i in enumerate(loc):
        loc_x, loc_y = loc_i
        pre_process = Lambda(crop, arguments={'loc_x':loc_x, 'loc_y':loc_y })(inputs)
        softmax_i, layer1_out = create_Model_single(pre_process,layers=[5, 5, 5])
        softmax_list += [softmax_i]
        layer1_out_list += [layer1_out]
    softmax = keras.layers.average(softmax_list*2)
    model = Model(inputs=inputs, outputs=softmax)
    return model, layer1_out_list

def yopo_adversary_generator(datagen, x, logits, m, n):
    old_generator = datagen.flow(x, logits, batch_size=args_batch_size)
    for x_batch, logits_batch in old_generator:
        np.random.seed(2019)
        eta = np.random.uniform(-args_eps, args_eps, x_batch.shape)
        x_new_batch = x_batch + eta
        for i in range(m + 1):
            yield x_new_batch, logits_batch
            if i == m:
                break
            # Add perturbation to inputs. loss_layer1 is only computed once.
            loss_layer1 = sess.run(loss_layer1_t_list, feed_dict={input_xs: x_new_batch, targets_ys: logits_batch,
                                                             sample_weights_ys: [1] * len(x_batch)})
            for j in range(n):
                loss_layer1_value = np.stack(loss_layer1, 0)
                grad = sess.run(yopo_grad_t, feed_dict={input_xs: x_new_batch, p_layer1_t: loss_layer1_value})
                # grad = sess.run(yopo_grad_t, feed_dict={input_xs: x_new_batch, p_layer1_t: loss_layer1})
                grad = np.sign(grad)
                x_new_batch += args_step_size * grad
                x_new_batch = np.clip(x_new_batch, x_batch - args_eps, x_batch + args_eps)
                x_new_batch = np.clip(x_new_batch, 0.0, 1.0)

# This is the (inputs, targets) generator for training
def adversary_generator(datagen, x, logits):
  old_generator = datagen.flow(x, logits, batch_size=args_batch_size)
  for x_batch, logits_batch in old_generator:
    x_new_batch = np.copy(x_batch)
    for i in range(args_step_num):
        grad = sess.run(grad_t, feed_dict={input_xs: x_new_batch, targets_ys: logits_batch, sample_weights_ys: [1] * len(x_batch)})
        grad = np.sign(grad)
        x_new_batch += args_step_size * grad
        x_new_batch = np.clip(x_new_batch, x_batch - args_eps, x_batch + args_eps)
        x_new_batch = np.clip(x_new_batch, 0.0, 1.0)
    yield x_new_batch, logits_batch

def lr_schedule(epoch):
  lr = 0.1
  if epoch > int(100 / args_lr_m):
    lr = 0.01
  elif epoch > int(150 / args_lr_m):
    lr = 0.001
  return lr

def sparse_loss_with_logits(y_train, pre_softmax_i):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=pre_softmax_i)

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

if __name__ == '__main__':
    from keras.datasets import mnist, cifar10, cifar100

    if 0:
        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        )
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
    else:
        sess= tf.Session()
    K.set_session(sess)
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())

    # Set parameters here
    args_step_size = 2.0 / 255.0
    args_eps = 8.0 / 255.0
    args_step_num = 7
    args_batch_size = 32
    args_yopo_m = 3  # m in YOPO-m-n
    args_yopo_n = 5  # n in YOPO-m-n
    args_lr_m = 3  # 1 if standard adversarial training, or use free_m or yopo_m
    num_classes = 10
    np.random.seed(2019)
    tf.set_random_seed(9102)
    # For reproduction
    np.random.seed(2019)
    tf.set_random_seed(9102)

    name_str = 'trans_Hope_Test_MNIST_YOPO'
    time_callback = TimeHistory()
    lr_scheduler = LearningRateScheduler(lr_schedule)
    #filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger(name_str + '_model_history_log.csv')
    callbacks = [time_callback, csv_logger]

    # Prepare data
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    logits_train = keras.utils.to_categorical(y_train, num_classes)
    logits_test = keras.utils.to_categorical(y_test, num_classes)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    train_datagen.fit(x_train)
    test_datagen = ImageDataGenerator()

    epochs = math.ceil(300 / args_yopo_m)
    opt = Adam(lr=1e-4, beta_1=0.5)

    model, layer1_out_list = create_Model(x_train[0].shape)

    #model = multi_gpu_model(model, gpus=4)
    model.compile(loss ='categorical_crossentropy', optimizer=opt,metrics=['categorical_accuracy'])

    # Standard adversarial
    input_xs = model.input
    output_ys = model.output
    targets_ys = model.targets[0]
    sample_weights_ys = model.sample_weights[0]
    loss_t = model.total_loss
    grad_t = K.gradients(loss_t, input_xs)[0]

    # YOPO
    yopo_grad_t = []
    loss_layer1_t_list = []
    p_layer1_t = K.placeholder([len(loc)] + layer1_out_list[0].get_shape().as_list(), dtype=tf.float32)
    for i, layer1_out in enumerate(layer1_out_list):
        loss_layer1_t = K.gradients(loss_t, layer1_out)  # gradient of loss w.r.t. the output of the first layer
        loss_layer1_t_list += loss_layer1_t
        hamilton_layer1_t = keras.layers.dot([Flatten()(p_layer1_t[i]), Flatten()(layer1_out)], axes=1)
        yopo_grad_t += [K.gradients(hamilton_layer1_t, input_xs)[0]]  # YOPO approximation of grad_t
    yopo_grad_t = keras.layers.average(yopo_grad_t*2,)


    sess.run(tf.global_variables_initializer())

    history = model.fit_generator(yopo_adversary_generator(train_datagen, x_train, logits_train, args_yopo_m, args_yopo_n),
                        validation_data=adversary_generator(test_datagen, x_test, logits_test),
                        validation_steps=math.ceil(len(x_test) / args_batch_size),
                        epochs=epochs, verbose=1, workers=1, callbacks= callbacks,
                        steps_per_epoch=math.ceil(len(x_train) / args_batch_size) * (args_yopo_m + 1))

    print('done')

    history.history['time'] = time_callback.times
    model.save(name_str + '_model.h5')

    with open(name_str + '_train_history', 'wb') as file:
        pickle.dump(history.history, file)

    with open(name_str + '_train_history', 'rb') as input_file:
        history = pickle.load(input_file)

    print(history.keys())

    plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name_str + '_accuracy.png')
    plt.clf()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name_str + '_loss.png')
    plt.clf()

    plt.plot(history['time'])
    plt.title('Model Training Time')
    plt.ylabel('Training Time (s)')
    plt.xlabel('Epoch')
    plt.savefig(name_str + '_time.png')
    plt.clf()
