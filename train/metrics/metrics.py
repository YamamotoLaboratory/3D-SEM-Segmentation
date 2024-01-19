import tensorflow as tf


def dice_coeff(y_true, y_pred, smooth = 1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    score = (2.0 * tf.reduce_sum(y_true_f * y_pred_f) + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def iou(y_true, y_pred):
    i = tf.reduce_sum(y_pred*y_true, axis=(1, 2))
    u = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - i
    return tf.reduce_mean(i / u)