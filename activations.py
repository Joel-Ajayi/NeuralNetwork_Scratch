import numpy as np
import tensorflow as tf

@tf.function
def relu_gradient(A):
    return tf.where(A > 0, tf.ones_like(A), tf.zeros_like(A))
