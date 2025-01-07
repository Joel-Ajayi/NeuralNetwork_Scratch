import tensorflow as tf

class Dropout:
  def __init__(self, drop_prob=0.5):
    self.keep_prob = tf.constant(1 - drop_prob)

  @property
  def variables(self):
    return []

  def init_batch_params(self, m, prev_units):
    self.d = tf.Variable(tf.zeros([prev_units, m]), trainable=False, dtype=tf.float32)

  
  @tf.function
  def forward(self, out_prev, eps, is_training=True):
    if not is_training:
      return out_prev

    self.d.assign(tf.cast(tf.random.uniform(tf.shape(out_prev),maxval=1) < self.keep_prob, dtype=tf.float32))

    out = tf.divide(tf.multiply(out_prev, self.d), self.keep_prob)
    return out
  
  @tf.function
  def backward(self, dout):
    dout_prev = tf.divide(tf.multiply(dout, self.d), self.keep_prob)
    return dout_prev

