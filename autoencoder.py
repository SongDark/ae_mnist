# coding:utf-8

from utils import *
from ops import * 
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

class AutoEncoder(BasicBlock):
    def __init__(self, len_latent=2, name='AE'):
        super(AutoEncoder, self).__init__(name)

        self.len_latent = len_latent
    
    def encode(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_encoder', reuse=reuse):
            
            net = lrelu(conv2d(x, 16, 4, 4, 2, 2, padding='SAME', name='c1'), name='l1') # 28x28x1->14x14x16
            net = lrelu(conv2d(net, 32, 4, 4, 2, 2, padding='SAME', name='c2'), name='l2') # 14x14x16->7x7x32
            net = tf.reshape(net, (-1, 7*7*32))

            code = dense(net, self.len_latent)

        return code
    
    def decode(self, code, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_decoder', reuse=reuse):

            net = lrelu(dense(code, 7*7*32, name='fc_1'), name='l1') # zdim -> 7x7x32
            net = tf.reshape(net, (-1, 7, 7, 32)) # 7x7x32

            net = lrelu(deconv2d(net, 16, 4, 4, 2, 2, padding='SAME', name='dc1'), name='l2') # 7x7x32->14x14x16
            net = tf.nn.sigmoid(deconv2d(net, 1, 4, 4, 2, 2, padding='SAME', name='dc2')) # 14x14x16->28x28x1

        return net

# z_dim = 100
# X = tf.ones(shape=(5, 28, 28, 1), dtype=tf.float32)

# AE = AutoEncoder(len_latent=z_dim)
# c = AE.encode(X)
# y = AE.decode(c)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(y).shape