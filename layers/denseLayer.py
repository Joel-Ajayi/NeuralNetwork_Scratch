from typing import Literal
import tensorflow as tf

class Dense:
    def __init__(
        self,
        units,
        activation:Literal["sigmoid", "softmax", "linear"]='linear',
        use_bias=True,
    ):
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.is_out_layer = self.activation in ["sigmoid", "softmax"]
    
    @property
    def variables(self):
      if self.use_bias:
        return [self.w, self.b]

      return [self.w]

    def init_batch_params(self, m, prev_units):
      self.out = tf.Variable(tf.zeros([self.units, m]), trainable=False, dtype=tf.float32)
      self.out_prev = tf.Variable(tf.zeros([prev_units, m]), trainable=False, dtype=tf.float32)
    
    def init_params(self,prev_units:int,optimizer:str):
        #     gamma: Scaling parameter (scalar) (n,1).
        #     beta: Shifting parameter (scalar) (n,1).
        #     m samples
        #     weights (n,n_prev)
        self.prev_units = prev_units
        self.opt = optimizer
        
        if self.activation in ['sigmoid']:
          # Xavier
          initializer = tf.keras.initializers.GlorotUniform()
        else:
        # He initialialization
          initializer = tf.keras.initializers.HeNormal()

        self.w = tf.Variable(initializer(shape=(self.units, self.prev_units)), trainable=True)
        self.dw = tf.Variable(tf.zeros_like(self.w), trainable=True)

        if self.use_bias:
          self.b = tf.Variable(tf.zeros([self.units, 1]), trainable=True, name='bias', dtype=tf.float32)
          self.db = tf.Variable(tf.zeros_like(self.b), trainable=True)

        # regularlization
        if optimizer == "momentum":
          self.beta1 = tf.constant(0.9)
          self.dVw = tf.Variable(tf.zeros_like(self.w), trainable=False)
          if self.use_bias:
            self.dVb = tf.Variable(tf.zeros_like(self.b), trainable=False)
        elif optimizer == "adam":
          self.beta1 = tf.constant(0.9)
          self.beta2 = tf.constant(0.999)

          self.dVw = tf.Variable(tf.zeros_like(self.w), trainable=False)
          self.dVw_c = tf.Variable(tf.zeros_like(self.w), trainable=False)

          self.dSw = tf.Variable(tf.zeros_like(self.w), trainable=False)
          self.dSw_c = tf.Variable(tf.zeros_like(self.w), trainable=False)

          if self.use_bias:
            self.dVb = tf.Variable(tf.zeros_like(self.b), trainable=False)
            self.dVb_c = tf.Variable(tf.zeros_like(self.b), trainable=False)

            self.dSb = tf.Variable(tf.zeros_like(self.b), trainable=False)
            self.dSb_c = tf.Variable(tf.zeros_like(self.b), trainable=False)

    
    @tf.function
    def forward(self, out_prev, eps, is_training=True):
        z = tf.matmul(self.w, out_prev)

        z = tf.add(z, self.b if self.use_bias else 0.0)

        if self.activation == "sigmoid":
            out = tf.nn.sigmoid(z)
        elif self.activation == "softmax":
            out = tf.nn.softmax(z, axis=0)
        else:
            out = z

        if is_training:
          self.out_prev.assign(out_prev)
          self.out.assign(out)

        tf.debugging.check_numerics(self.out, "NaN detected in activations")
        return out
    
    @tf.function
    def backward_out(self, Y):
        m = tf.cast(Y.shape[1], tf.float32) 

        dz = (1.0 / m) * tf.subtract(self.out, Y)
        self.dw.assign(tf.matmul(dz, tf.transpose(self.out_prev)))
        dout_prev = tf.matmul(tf.transpose(self.w), dz)
        if self.use_bias:
          self.db.assign(tf.reduce_sum(dz, axis=1, keepdims=True))       
        return dout_prev
    
    @tf.function
    def backward(self, dout):
        self.dw.assign(tf.matmul(dout, tf.transpose(self.out_prev)))
        if self.use_bias:
          self.db.assign(tf.reduce_sum(dout, axis=1, keepdims=True))
        dout_prev = tf.matmul(tf.transpose(self.w), dout)
        
        return dout_prev
    
    @tf.function
    def update(self, learning_rate, eps, t):
        # apply optimisers
        self.optimize(t)

        if self.opt == "gd":
            self.w.assign_sub(learning_rate * self.dw)
            if self.use_bias:
                # Update bias only in the output layer
                self.b.assign_sub(learning_rate * self.db)
        elif self.opt == "momentum":
            self.w.assign_sub(learning_rate * self.dVw)

            if self.use_bias:
                # Update bias only in the output layer
                self.b.assign_sub(learning_rate * self.dVb)
        elif self.opt == "adam":
            self.w.assign_sub(learning_rate * tf.divide(self.dVw_c, tf.add(eps, tf.sqrt(self.dSw_c))))
            if self.use_bias:
                self.b.assign_sub(learning_rate * tf.divide(self.dVb_c, tf.add(eps, tf.sqrt(self.dSb_c))))
    
    @tf.function
    def optimize(self, t):
        if self.opt == "momentum":
            self.optimize_momentum()
        elif self.opt == "adam":
            self.optimize_adam(t)
    
    @tf.function
    def optimize_momentum(self):
        self.dVw.assign((self.beta1 * self.dVw) + ((1.0 - self.beta1) * self.dw))
        if self.use_bias:
            self.dVb.assign((self.beta1 * self.dVb) + ((1.0 - self.beta1) * self.db))
    
    @tf.function
    def optimize_adam(self, t):
        beta1 = self.beta1
        beta2 = self.beta2

        # Moving average of the gradients and corrections
        self.dVw.assign((beta1 * self.dVw) + ((1.0 - beta1) * self.dw))
        self.dVw_c.assign(self.dVw / (1.0 - beta1**t))

        if self.use_bias:
            self.dVb.assign((beta1 * self.dVb) + ((1 - beta1) * self.db))
            self.dVb_c.assign(self.dVb / (1.0 - beta1**t))

        # Moving average of the squared gradients. for weights, beta and gamma
        self.dSw.assign((beta2 * self.dSw) + ((1.0 - beta2) * (self.dw**2)))
        self.dSw_c.assign(self.dSw / (1.0 - beta2**t))

        if self.use_bias:
            self.dSb.assign((beta2 * self.dSb) + ((1.0 - beta2) * (self.db**2)))
            self.dSb_c.assign(self.dSb / (1.0 - beta2**t))

    
