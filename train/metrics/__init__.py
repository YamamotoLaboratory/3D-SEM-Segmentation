from .metrics import dice_coeff, IoU

class MetricsValue(object):
    
    def __init__(self, name, metrics):

        self.data = self.set_data(metrics)
        self.func = self.set_func(name)
        self.value = None

    def set_data(self, metrics):
        import tensorflow as tf
        if metrics == 'mean':
            return tf.keras.metrics.Mean()
        elif metrics == 'recall':
            return tf.keras.metrics.Recall()
        elif metrics == 'precision':
            return tf.keras.metrics.Precision()
        
    def set_func(self, name):
        if name == 'dice_coeff':
            return dice_coeff
        elif name == 'IoU':
            return IoU
        else:
            return None
    
    def calc(self, masks, logits):
        if self.func == None:
            self.value = [masks, logits]
        else:
            self.value = self.func(masks, logits)

    def update(self, ):
        if self.func == None:
            self.data.update_state(self.value[0], self.value[1])
        else:
            self.data.update_state(self.value)
