import tensorflow as tf
import keras
from typing import Self
import os
import absl
from customs.attention import AttentionGate

#disable useless logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 for just errors or 2 for warning and errors
tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)

class ModelBuilder:
    
    def __init__(self, input_shape:tuple[int, int, int]):
        self._input_shape = input_shape
        
        self.set_n_coef(16)
        self.set_kernel_init('he_normal')
        self.set_conv_activation('leaky_relu')
        self.unset_compile()
    
    def unset_compile(self):
        self._compile = False
        
        if hasattr(self, '_loss'):
            delattr(self, '_loss')
        
        if hasattr(self, '_optimizer'):
            delattr(self, '_optimizer')
        
        if hasattr(self, '_metrics'):
            delattr(self, 'metrics')
    
    def set_compile(self, optimizer:keras.optimizers.Optimizer|str = 'adam', 
                    loss=keras.losses.dice, metrics=[]) -> Self:
        self._optimizer = optimizer
        self._loss = loss,
        self._metrics = metrics
        self._compile = True
        
        return self
    
    def set_n_coef(self, n_coef:int) -> Self:
        self._n_coef = n_coef
        return self
        
    def set_kernel_init(self, kernel_init:str) -> Self:
        self._kernel_init = kernel_init
        return self
    
    def set_conv_activation(self, conv_activation) -> Self:
        self._conv_activation = conv_activation
        return self
    
    def build(self):
        
        base_model = keras.applications.InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=self._input_shape    
        )
        
        base_model.trainable = False
        
        input_ = base_model.layers[0].output

        #region encoder 
        
        #block 1
        sk1 = base_model.layers[1].output
        sk1 = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(sk1)
        sk1 = keras.layers.Conv2DTranspose(32, 2, 2)(sk1)
        
        #block 2
        sk2 = base_model.layers[3].output
        sk2 = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(sk2)

        #block 3
        sk3 = base_model.layers[13].output
        sk3 = keras.layers.ZeroPadding2D()(sk3)
        
        #block 4
        sk4 = base_model.layers[274].output
        sk4 = keras.layers.ZeroPadding2D(9)(sk4)        
        sk4 = keras.layers.Conv2D(256, 1, padding='same')(sk4)
        
        #endregion
        
        #region bridge (Bottle neck)
        
        brg = base_model.layers[501].output
        brg = keras.layers.ZeroPadding2D()(brg)
        
        #endregion
        
        #region decoder
        
        d1 = self._create_upsample_block(512, 2)(brg, sk4)
        d2 = self._create_upsample_block(256, 2)(d1, sk3)
        d3 = self._create_upsample_block(128, 2)(d2, sk2)
        d4 = self._create_upsample_block(64, 2)(d3, sk1)
        
        #endregion        
        
        #region output
        
        o = keras.layers.Conv2D(1, 1, activation='sigmoid')(d4)
        
        #endregion
        
        model = keras.models.Model(inputs=input_, outputs=o)
        
        if self._compile:
            model.compile(self._optimizer, self._loss, metrics=[*self._metrics, ModelBuilder._dice_metric])
        
        return model
    
    @staticmethod
    def _dice_metric(y_true, y_pred):
        return 1 - keras.losses.dice(y_true, y_pred)
    
    # block 6 in irv2-net paper
    def _create_block6(self, filters):
        
        def block6(input):
            b = keras.layers.Conv2D(filters, (1, 1), padding='same')(input)
            b = keras.layers.BatchNormalization()(b)
            b = keras.layers.Activation('relu')(b)
            b = keras.layers.Conv2D(filters, (1, 1), padding='same')(input)
            b = keras.layers.BatchNormalization()(b)
            b = keras.layers.Activation('relu')(b)
            
            return b
        
        return block6
    
    def _create_upsample_block(self, filters, kernel_size):
        
        def upsample_block(input, skip_connection):
            b = keras.layers.Conv2DTranspose(filters, kernel_size, 2, activation='relu')(input)
            skip_connection = AttentionGate(filters, 'he_normal','relu')(skip_connection, b)
            b = keras.layers.concatenate([b, skip_connection])
            b = self._create_block6(filters)(b)
            
            return b
        
        return upsample_block