import tensorflow as tf
import tensorflow_addons as tfa


class Optimizer:
    
    def __init__(self, opt, initial_lr, wd = 0):
        self.opt = opt
        self.initial_lr = initial_lr
        self.wd = wd
    def get_optimizer(self, ):
        if self.opt == 'SGD':
            return tf.keras.optimizers.SGD(learning_rate = self.initial_lr)
        elif self.opt == 'Adam':
            return tf.keras.optimizers.Adam(learning_rate = self.initial_lr)
        elif self.opt == 'RAdam':
            return tfa.optimizers.RectifiedAdam(learning_rate = self.initial_lr, weight_decay = self.wd)