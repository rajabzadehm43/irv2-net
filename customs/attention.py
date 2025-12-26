import tensorflow as tf
from tensorflow import keras
import numpy as np

class AttentionGate(keras.layers.Layer):
    def __init__(self, inter_channel:int, kernel_initializer='glorot_uniform', activation='relu', **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.inter_channel = inter_channel

        # come from skip connection
        self.x_conv = keras.layers.Conv2D(self.inter_channel, 1, padding='same', kernel_initializer=kernel_initializer)

        # come from prev decoder layer
        self.g_conv = keras.layers.Conv2D(self.inter_channel, 1, padding='same', kernel_initializer=kernel_initializer)

        self.add = keras.layers.Add()

        if type(activation) is str:
            self.activation = keras.layers.Activation(activation)
        else:
            self.activation = activation

        self.att_conv = keras.layers.Conv2D(1, 1, padding='same', kernel_initializer=kernel_initializer)
        self.sigmoid = keras.activations.sigmoid
        self.b_norm = keras.layers.BatchNormalization()

    def call(self, x, g):
        """
        Args:
            x: output from skip connection
            g: output from prev decoder layer

        Returns:
            Attention map
        """
        x = self.x_conv(x)
        g = self.g_conv(x)

        res = self.add([x, g])
        res = self.activation(res)
        
        a_map = self.att_conv(res)
        a_map = self.sigmoid(a_map)

        res = tf.multiply(x, a_map)
        return self.b_norm(res)

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(CustomCallback, self).__init__(**kwargs)
        self.max_val_accuracy = {'epoch': 0, 'value': 0}
        self.max_train_accuracy = {'epoch': 0, 'value': 0}

    def on_train_begin(self, logs=None):
        self.max_val_accuracy = {'epoch': 0, 'value': 0}
        self.max_train_accuracy = {'epoch': 0, 'value': 0}

    def on_epoch_end(self, epoch, logs=None):
        
        epoch += 1
        val_dice_coef = logs['val_dice_coef']
        train_dice_coef = logs['dice_coef']

        if self.max_val_accuracy['value'] < val_dice_coef:
            self.max_val_accuracy = {'epoch': epoch, 'value': val_dice_coef}

        if self.max_train_accuracy['value'] < train_dice_coef:
            self.max_train_accuracy = {'epoch': epoch, 'value': train_dice_coef}

    def on_train_end(self, logs=None):

        max_val_epoch = self.max_val_accuracy.get('epoch')
        max_val_value = self.max_val_accuracy.get('value')

        max_train_epoch = self.max_train_accuracy.get('epoch')
        max_train_value = self.max_train_accuracy.get('value')

        print()
        print(f'max val dice coef is "{max_val_value:.4f}" in epoch "{max_val_epoch}"')
        print(f'max train dice coef is "{max_train_value:.4f}" in epoch "{max_train_epoch}"')