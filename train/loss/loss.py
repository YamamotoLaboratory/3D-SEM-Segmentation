import tensorflow as tf

def dice_loss(y_true, y_pred, smooth = 1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    score = (2.0*tf.reduce_sum(y_true_f * y_pred_f) + smooth)/(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    loss = 1 - score
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def cross_entropy(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)