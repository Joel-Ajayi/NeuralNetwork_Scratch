from activations import relu_gradient
import tensorflow as tf
import numpy as np

class Relu():
    def __init__(self):
      self.alpha = 0.1

    def init_batch_params(self, m, prev_units):
      self.out = tf.Variable(tf.zeros([prev_units, m]), trainable=False, dtype=tf.float32)

    @property
    def variables(self):
      return []
    
    @tf.function
    def forward(self, out_prev, eps:float, is_training=True):
      out = tf.nn.relu(out_prev)
      if is_training:
        self.out.assign(out)

      return out
    
    @tf.function
    def backward(self, dout):
      dout_dout_prev = relu_gradient(self.out)
      dout_prev = tf.multiply(dout, dout_dout_prev)

      return dout_prev


      

    