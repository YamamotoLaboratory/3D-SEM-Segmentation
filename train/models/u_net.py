from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPooling2D,
    ReLU,
    concatenate,
)


class Unet:
    def __init__(self, input_shape, filters, depth, input_channel, class_num, logger, kernel_size=(4, 4)):
        self.SHAPE = input_shape
        self.FILTERS = filters
        self.DEPTH = depth
        self.INPUT_CHANNEL = input_channel
        self.CLASS_NUM = class_num
        self.KERNEL_SIZE = kernel_size
        self.logger = logger
        self.logger.info('Model: U-Net, Input Size: {}x{}'.format(input_shape[0], input_shape[1]))
        
    def _down_conv_block(self, x, filters, apply_relu=True, apply_batchnormalization=True):
        x = Conv2D(filters, kernel_size = self.KERNEL_SIZE, strides = 1, padding = 'same', kernel_initializer = 'he_normal', use_bias = False)(x)
        if apply_relu:
            x = ReLU()(x)
        if apply_batchnormalization:
            x = BatchNormalization()(x)
        return x
    
    def _up_conv_block(self, x, filters, apply_relu=True):
        x = Conv2D(filters*2, kernel_size = self.KERNEL_SIZE, strides = 1, padding = 'same', kernel_initializer = 'he_normal', use_bias = False)(x)
        if apply_relu:
            x = ReLU()(x)
        return x
    
    
    def model_call(self, x):
        skip = []
        for i in range(self.DEPTH):
            if i == 0:
                x = self._down_conv_block(x, self.FILTERS*2**(i))
            elif i == self.DEPTH-1:
                x = self._down_conv_block(x, self.FILTERS*2**(i))
            else:
                x = self._down_conv_block(x, self.FILTERS*2**(i))
            skip.append(x)
            x = MaxPooling2D([2, 2])(x)
        x = self._down_conv_block(x, self.FILTERS*2**(self.DEPTH))
        for i in range(self.DEPTH, 0, -1):
            x = Conv2DTranspose(self.FILTERS*2**(i-1), kernel_size = 2, strides = 2, padding = 'same', kernel_initializer = 'he_normal', use_bias = False)(x)
            x = concatenate([x, skip[i-1]], -1)
            x = self._up_conv_block(x, self.FILTERS*2**(i-2))
        
        x = Conv2D(self.CLASS_NUM, kernel_size = self.KERNEL_SIZE, strides = 1, padding = 'same', kernel_initializer = 'he_normal', use_bias = False)(x)
        return sigmoid(x)
        
    def get_model(self, ):
        x = Input(shape = [self.SHAPE[0], self.SHAPE[1], self.INPUT_CHANNEL])
        self.get_static_model = Model(inputs = [x], outputs = self.model_call(x))
        self.logger.info('U-Net作成完了')
        return self.get_static_model