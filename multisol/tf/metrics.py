import tensorflow as tf

eps = tf.constant(1e-6, dtype=tf.float32)

@tf.function
def accuracy(TN, FP, FN, TP):
    return (TP + TN) / (TN + FP + FN + TP + eps)

@tf.function
def precision(TN, FP, FN, TP):
    return TP / (FP + TP + eps)

@tf.function
def recall(TN, FP, FN, TP):
    return TP / (FN + TP + eps)

@tf.function
def specificity(TN, FP, FN, TP):
    return TN / (FP + TN + eps)

@tf.function
def f1_score_fun(TN, FP, FN, TP):
    p = precision(TN, FP, FN, TP)
    r = recall(TN, FP, FN, TP)
    return 2 * p * r / (p + r + eps)

@tf.function
def tss(TN, FP, FN, TP):
    return recall(TN, FP, FN, TP) + specificity(TN, FP, FN, TP) - 1

@tf.function
def gmean(TN, FP, FN, TP):
    sens = tf.cond(
        tf.greater(TP + FN, 0),
        lambda: TP / (TP + FN),
        lambda: tf.constant(0.0, dtype=TP.dtype)
    )
    spec = tf.cond(
        tf.greater(TN + FP, 0),
        lambda: TN / (TN + FP),
        lambda: tf.constant(0.0, dtype=TN.dtype)
    )
    return tf.sqrt(sens * spec)