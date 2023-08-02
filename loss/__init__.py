from .loss import dice_loss, bce_dice_loss, cross_entropy

class LossValue(object):

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
        if name == 'dice_loss':
            return dice_loss
        elif name == 'bce_dice_loss':
            return bce_dice_loss
        elif name == 'cross_entropy':
            return cross_entropy
        else:
            return None
        
    def calc(self, masks, logits):
        if self.func == None:
            self.value = [masks, logits]
        else:
            self.value = self.func(masks, logits)

    def update(self, ):
        if self.func != None:
            self.data.update_state(self.value)
        else:
            self.data.update_state(self.value[0], self.value[1])