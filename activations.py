import numpy as np
import tensorflow as tf

@tf.function
def relu_gradient(A, alpha):
    return tf.where(A > 0, tf.ones_like(A), tf.fill(tf.shape(A), alpha))
