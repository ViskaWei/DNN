import os
import numpy as np
import pandas as pd
import logging
# import signal
import h5py
import datetime
# from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr
import tensorflow.keras.activations as ka
from tensorflow.keras import Sequential as ks
from tensorflow.keras import optimizers as ko
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")
# import tensorflow.python.keras.optimizer_v2 as ko



class DNN(object):
    def __init__(self, input_dim=20,  hidden_dims=[], reg1=None,\
                        dp=0., loss='mse', lr=0.01, opt='adam', name=''):
        # super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = 5
        self.input = keras.Input(shape = (self.input_dim, ), name='input')
        self.output = keras.Input(shape = (self.output_dim, ), name='out')
        self.hidden_dims = np.array(hidden_dims)
        self.units = self.get_units()

        self.reg1 = reg1
        self.dp   = dp

        self.model = None
        self.lr = lr
        self.opt = self.get_opt(opt)
        self.loss = loss
        self.name = self.get_name(name)
        self.log_dir = "logs/fit/" + self.name

        self.callbacks = [
            EarlyStopping(monitor='loss', patience=6),
            ReduceLROnPlateau('loss',patience=3, min_lr=0., factor=0.1),
            # TensorBoard(log_dir=self.log_dir)
            # TensorBoard(log_dir=self.log_dir, histogram_freq=1)
            MyCallback()
        ]

    def get_opt(self, opt):
        if opt == 'adam':
            return ko.Adam(learning_rate=self.lr, decay=1e-6)
        if opt == 'sgd':
            return ko.SGD(learning_rate=self.lr, momentum=0.9)
        else:
            raise 'optimizer not working'

    def get_name(self, name):
        lr_name = -np.log10(self.lr)
        out_name = f'{self.loss}_lr{lr_name}_h{len(self.hidden_dims)}'
        if self.dp != 0:
            out_name = out_name + f'dp{self.dp}_'
        t = datetime.datetime.now().strftime("%m%d-%H%M%S")
        out_name = out_name + name + '_' + t
        return out_name.replace('.', '')


    def fit(self, x_train, y_train, ep=50, batch=512):
        self.model.fit(x_train, y_train, 
                    epochs=ep, 
                    batch_size=batch, 
                    validation_split=0.2, 
                    callbacks=self.callbacks,
                    shuffle=True,
                    verbose=2
                    )
    
    def set_model_shapes(self, input_shape, latent_size):
        self.input_shape = (input_shape[1], )
        self.latent_size = latent_size

    def get_units(self):
        if self.hidden_dims.size == 0:
            return [self.input_dim //2, self.input_dim //4, self.input_dim //8 ] 
        self.hidden_dims = self.hidden_dims[self.hidden_dims > self.output_dim]
        units = [self.input_dim, *self.hidden_dims, self.output_dim]
        print(units)
        return units 

    def build_dnn(self):
        x = self.input
        for ii, unit in enumerate(self.units[1:]):
            name = 'l' + str(ii)
            x = self.add_dense_layer(unit, dp_rate=self.dp, reg1=self.reg1, name=name)(x)
        self.model = keras.Model(self.input, x, name="dnn")


    def build_model(self):
        self.build_dnn()
        self.model.compile(
                loss=self.loss,
                optimizer=self.opt,
                # metrics=['acc'],
                metrics=[MeanSquaredError()]
            )


    def add_dense_layer(self, unit, dp_rate=0., reg1=None, name=None):
        if reg1 is not None:
            kl1 = tf.keras.regularizers.l1(reg1)
        else:
            kl1 = None

        layer = ks([kl.Dense(unit, kernel_regularizer=kl1, name=name),
                    # kl.BatchNormalization(),
                    kl.LeakyReLU(),
                    kl.Dropout(dp_rate)
                    # keras.activations.tanh()
                    ])
        return layer

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 1:
            print(epoch, self.model.__dict__)
        else:
            pass
