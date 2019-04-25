# coding:utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import *
from autoencoder import AutoEncoder
from variational_autoencoder import Variational_AutoEncoder
from datamanager import datamanager_mnist

class Trainer(BasicTrainFramework):
    
    def __init__(self, batch_size, version=None, gpu='0'):
        super(Trainer, self).__init__(batch_size, version=version, gpu=gpu)

        self.data = datamanager_mnist(train_ratio=0.8, fold_k=None, expand_dim=True, norm=True)
        self.sample_data = self.data(self.batch_size, 'test', var_list=['data'])

        self.emb_dim = 32

        self.build_placeholder()

        if version=='AE':
            self.autoencoder = AutoEncoder(len_latent=self.emb_dim, name='AE')
            self.build_ae_network()
        elif version == 'VAE':
            self.autoencoder = Variational_AutoEncoder(len_latent=self.emb_dim, name='VAE')
            self.build_vae_network()
        
        self.build_optimizer()
        self.build_summary()

        self.build_dirs()
        self.build_sess()

    def build_placeholder(self):
        self.source = tf.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)
        self.target = tf.placeholder(shape=(self.batch_size, 28, 28, 1), dtype=tf.float32)

    
    def build_ae_network(self):
        self.code = self.autoencoder.encode(self.source, is_training=True, reuse=False)
        self.cyc = self.autoencoder.decode(self.code, is_training=True, reuse=False)

        # only for test
        code_test = self.autoencoder.encode(self.source, is_training=False, reuse=True)
        self.cyc_test = self.autoencoder.decode(code_test, is_training=False, reuse=True)
    
    def build_vae_network(self):
        self.mean_code, self.std_code = self.autoencoder.encode(self.source, is_training=True, reuse=False)
        gaussian_noise = tf.random_normal(tf.shape(self.mean_code), 0.0, 1.0, dtype=tf.float32)
        self.cyc = self.autoencoder.decode(gaussian_noise, self.mean_code, self.std_code, is_training=True, reuse=False)

        # only for test
        mean_code_test, std_code_test = self.autoencoder.encode(self.source, is_training=False, reuse=True)
        self.cyc_test = self.autoencoder.decode(gaussian_noise, mean_code_test, std_code_test, is_training=False, reuse=True)

    def build_optimizer(self):
        self.cyc_loss = tf.reduce_mean(tf.squared_difference(self.cyc, self.target))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.cyc_solver = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.cyc_loss, var_list=self.autoencoder.vars)
    
    def build_summary(self):
        self.cycloss_sum = tf.summary.scalar("cycloss", self.cyc_loss)
        self.sumstr = tf.summary.merge([self.cycloss_sum])

    def sample(self, epoch):
        def plot(imgs, save_path):
            tmp = [[] for _ in range(5)]
            for i in range(5):
                for j in range(5):
                    tmp[i].append(imgs[i*5+j])
                tmp[i] = np.concatenate(tmp[i], 1)
            tmp = np.concatenate(tmp, 0)
            plt.imshow(tmp[:,:,0], cmap=plt.cm.gray)
            plt.savefig(save_path)
            plt.clf()
        
        print "sample at epoch {}".format(epoch)
        feed_dict = {self.source : self.sample_data['data']}

        cyc = self.sess.run(self.cyc_test, feed_dict=feed_dict)

        plot(cyc, os.path.join(self.fig_dir, "cyc_{}.png".format(epoch)))

        if epoch == 0:
            plot(self.sample_data['data'], os.path.join(self.fig_dir, "ori.png"))
    
    def train(self, epoches=1):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            self.data.shuffle_train(seed=epoch)

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx

                src = self.data(self.batch_size, 'train', var_list=['data'])

                feed_dict = {self.source:src['data'], self.target:src['data']}

                self.sess.run(self.cyc_solver, feed_dict=feed_dict)

                if cnt % 25 == 0:
                    cyc_loss, sumstr = self.sess.run([self.cyc_loss, self.sumstr], feed_dict=feed_dict)
                    print self.version + " Epoch [%3d/%3d] Iter [%3d] loss=%.4f" % (epoch, epoches, idx, cyc_loss)
                    self.writer.add_summary(sumstr, cnt)
            
            if epoch % 50 == 0:
                self.sample(epoch)
        
        self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, "model.ckpt"))

def ae():
    trainer = Trainer(64, version='AE', gpu='2')
    trainer.train(500)

def vae():
    trainer = Trainer(64, version='VAE', gpu='2')
    trainer.train(500)

def plot_loss():
    def filtering(seq, window=3):
        seq = np.array(seq)[:,None]
        d = window // 2
        res = []
        for i,j in zip([0] * d + range(len(seq) - d), range(d, len(seq)) + [len(seq)] * d):
            res.append(np.mean(seq[i:j+1, :], axis=0))
        return np.asarray(res)
    aeloss = event_reader("save/AE/logs", names=['cycloss'])
    vaeloss = event_reader("save/VAE/logs", names=['cycloss'])
    plt.plot(filtering(aeloss['cycloss'][1000:], 25), linewidth=1, color='g', label='ae')
    plt.plot(filtering(vaeloss['cycloss'][1000:], 25), linewidth=1, color='r', label='vae')
    plt.legend()
    plt.savefig("save/loss.png")
    
ae()
vae()
plot_loss()



