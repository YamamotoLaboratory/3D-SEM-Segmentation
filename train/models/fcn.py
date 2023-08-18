import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.activations import sigmoid

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2DTranspose,
    Conv2D,
    MaxPooling2D,
    Input,
    ReLU,
    Add,
    Dropout,
)

class FCN:
    
    def __init__(self, input_shape, input_channel, class_num, logger, up='32s', backbone='VGG-16'):
        self.SHAPE = input_shape
        self.INPUT_CHANNEL = input_channel
        self.CLASS_NUM = class_num
        self.logger = logger
        self.up = up
        self.backbone = backbone
        self.logger.info('Model: FCN-{}, Input Size: {}x{}, Backbone: {}'.format(up, input_shape[0], input_shape[1], backbone))
    
    def vgg_Conv(self, x, channels):
        x = Conv2D(channels, kernel_size = (3, 3), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = ReLU()(x)
        return x
    
    def res_input_block(self, x):
        x = Conv2D(64, kernel_size = (7, 7), strides = (2, 2), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding = 'same')(x)
        return x
    
    def _res_block(self, inputs, input_channel, output_channel, slides = (1, 1), dilation_rate=(1, 1) ,skip_connection = True, projection = True, get_afterMP=False):
        input_depth = inputs.get_shape().as_list()[3]
        
        x = Conv2D(input_channel, kernel_size = (1, 1), strides = slides, dilation_rate = dilation_rate, kernel_initializer="he_normal", padding = 'same', use_bias = False)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        if get_afterMP:
            p = x
        
        x = Conv2D(input_channel, kernel_size = (3, 3), strides =(1, 1), dilation_rate = dilation_rate, kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        x = Conv2D(output_channel, kernel_size = (1, 1), strides =(1, 1), dilation_rate = dilation_rate, kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        
        if input_depth != output_channel:
            if projection:
                inputs = Conv2D(filters = output_channel, kernel_size = 1, strides = 1, kernel_initializer="he_normal", padding = 'same', use_bias = False)(inputs)
                
            else:
                inputs = tf.pad(inputs, [[0,0], [0,0], [0,0], [0, output_channel - input_depth]])
                
                
        if skip_connection:
            x = Add()([inputs, x])
            x = ReLU()(x)
        else:
            x = ReLU()(x)
        if get_afterMP:
            return x, p
        else:
            return x
    
    def vgg_last_Conv(self, x, channels):
        x = Conv2D(channels, kernel_size = (7, 7), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(channels, kernel_size = (1, 1), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        return x
    
    def decoder32(self, x):
        x = Conv2D(self.CLASS_NUM, kernel_size = (1, 1), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = Conv2DTranspose(self.CLASS_NUM, kernel_size = (64, 64), strides = (32, 32), padding = 'same', use_bias = False)(x)
        x = sigmoid(x)
        return x
    
    def decoder16(self, x, p4):
        x = Conv2D(self.CLASS_NUM, kernel_size = (1, 1), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = Conv2DTranspose(self.CLASS_NUM, kernel_size = (4, 4), strides = (2, 2), padding = 'same', use_bias = False)(x)
        p4 = Conv2D(self.CLASS_NUM, kernel_size = (1, 1), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(p4)
        x = Add()([x, p4])
        x = Conv2DTranspose(self.CLASS_NUM, kernel_size = (32, 32), strides = (16, 16), padding = 'same', use_bias = False)(x)
        x = sigmoid(x)
        return x
        
    def decoder8(self, x, p4, p3):
        x = Conv2D(self.CLASS_NUM, kernel_size = (1, 1), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = Conv2DTranspose(self.CLASS_NUM, kernel_size = (4, 4), strides = (2, 2), padding = 'same', use_bias = False)(x)
        p4 = Conv2D(self.CLASS_NUM, kernel_size = (1, 1), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(p4)
        x = Add()([x, p4])
        
        p3 = Conv2D(self.CLASS_NUM, kernel_size = (1, 1), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(p3)
        x = Conv2D(self.CLASS_NUM, kernel_size = (1, 1), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = Conv2DTranspose(self.CLASS_NUM, kernel_size = (4, 4), strides = (2, 2), padding = 'same', use_bias = False)(x)
        x = Add()([x, p3])
        
        x = Conv2DTranspose(self.CLASS_NUM, kernel_size = (16, 16), strides = (8, 8), padding = 'same', use_bias = False)(x)
        x = sigmoid(x)
        return x
    
    def alex_net(self, x):
        x = Conv2D(96, kernel_size = (11, 11), strides = (4, 4), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = ReLU()(x)
        x = MaxPooling2D([3, 3], strides = (2, 2), padding = 'same')(x)
        p3 = x
        x = Conv2D(256, kernel_size = (5, 5), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = ReLU()(x)
        x = MaxPooling2D([3, 3], strides = (2, 2), padding = 'same')(x)
        p4 = x
        x = Conv2D(384, kernel_size = (3, 3), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = ReLU()(x)
        x = Conv2D(384, kernel_size = (3, 3), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = ReLU()(x)
        x = Conv2D(256, kernel_size = (3, 3), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = ReLU()(x)
        x = MaxPooling2D([5, 5], strides = (2, 2), padding = 'same')(x)
        
        x = Conv2D(4096, kernel_size = (7, 7), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(4096, kernel_size = (1, 1), strides = (1, 1), kernel_initializer="he_normal", padding = 'same', use_bias = False)(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        
        return x, p3, p4
    
    def vgg_16(self, x):
        level = [2, 2, 3, 3, 3]
        
        for i in range(level[0]):
            x = self.vgg_Conv(x, 64)
        x = MaxPooling2D([2, 2])(x)
        
        for i in range(level[1]):
            x = self.vgg_Conv(x, 128)
        x = MaxPooling2D([2, 2])(x)
        
        for i in range(level[2]):
            x = self.vgg_Conv(x, 256)
        x = MaxPooling2D([2, 2])(x)
        p3 = x
        
        for i in range(level[3]):
            x = self.vgg_Conv(x, 512)
        x = MaxPooling2D([2, 2])(x)
        p4 = x
        
        for i in range(level[3]):
            x = self.vgg_Conv(x, 512)
        x = MaxPooling2D([2, 2])(x)
        
        
        x = self.vgg_last_Conv(x, 4096)
        
        return x, p3, p4
    
    def resnet50(self, x):
        x = self.res_input_block(x)
        for i in range(3):
            x = self._res_block(x, 64, 256, )
        for i in range(4):
            if i == 0:
                x, p3 = self._res_block(x, 128, 512, slides = (2, 2), skip_connection = False, get_afterMP=True)
            else:
                x = self._res_block(x, 128, 512)
        for i in range(6):
            if i == 0:
                x, p4 = self._res_block(x, 256, 1024, slides = (2, 2), skip_connection = False, get_afterMP=True)
            else:
                x = self._res_block(x, 256, 1024)
        for i in range(3):
            if i == 0:
                x = self._res_block(x, 512, 2048, slides = (2, 2), skip_connection = False)
            else:
                x = self._res_block(x, 512, 2048)
        return x, p3, p4
    
    def model_call(self, x):
        if self.backbone == 'VGG-16':
            x, p3, p4 = self.vgg_16(x)
        elif self.backbone == 'AlexNet':
            x, p3, p4 = self.alex_net(x)
        else:
            x, p3, p4 = self.resnet50(x)
        if self.up == '32s':
            x = self.decoder32(x)
        elif self.up == '16s':
            x = self.decoder16(x, p4)
        else:
            x = self.decoder8(x, p4, p3)
        return x
        
    def get_model(self, ):
        x = Input(shape = [self.SHAPE[0], self.SHAPE[1], self.INPUT_CHANNEL])
        self.get_static_model = Model(inputs = [x], outputs = self.model_call(x))
        self.logger.info('FCN-{}作成完了'.format(self.up))
        return self.get_static_model