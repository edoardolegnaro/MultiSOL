import numpy as np
import tensorflow as tf

def F_unif(x):
    return x

pi_tf = tf.constant(np.pi, dtype=tf.float32)

def F_cos(x, mu, delta):
    pi = tf.constant(np.pi, dtype=x.dtype)
    inner_val = 0.5 * (1 + (x - mu) / delta + (1 / pi) * tf.sin(pi * (x - mu) / delta))
    return tf.where(x < mu - delta,
                    tf.zeros_like(x),
                    tf.where(x > mu + delta,
                             tf.ones_like(x),
                             inner_val))

def pdf_F_cos(x, mu, delta):
    pi = tf.constant(np.pi, dtype=x.dtype)
    return 1 / (2 * delta) * (1 + tf.cos(pi * (x - mu) / delta))