import tensorflow as tf


def dice_coeff(y_true, y_pred, smooth = 1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    score = (2.0 * tf.reduce_sum(y_true_f * y_pred_f) + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def iou(y_true, y_pred):
    # しきい値を適用してバイナリマスクを得るか、既にバイナリマスクであることを確認する
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_bin = tf.cast(y_true > 0.5, tf.float32)

    # インターセクションとユニオンの計算
    i = tf.reduce_sum(y_pred_bin * y_true_bin, axis=(1, 2))
    u = tf.reduce_sum(tf.cast(y_pred_bin + y_true_bin > 0, tf.float32), axis=(1, 2))

    # 0で割ることを避けるために安全な割り算を使用
    return tf.reduce_mean(i / (u + tf.keras.backend.epsilon()))