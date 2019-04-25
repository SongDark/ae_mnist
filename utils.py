# coding:utf-8

import tensorflow as tf 
import numpy as np 
import os 
import re

class BasicBlock(object):
    def __init__(self, name):
        self.name = name
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

class BasicTrainFramework(object):
	def __init__(self, batch_size, version, gpu='0'):
		self.batch_size = batch_size
		self.version = version
		os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	def build_dirs(self):
		self.log_dir = os.path.join('save', self.version, 'logs') 
		self.model_dir = os.path.join('save', self.version, 'checkpoints')
		self.fig_dir = os.path.join('save', self.version, 'figs')
		for d in [self.log_dir, self.model_dir, self.fig_dir]:
			if (d is not None) and (not os.path.exists(d)):
				print "mkdir " + d
				os.makedirs(d)
	
	def build_sess(self):
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def load_model(self, checkpoint_dir=None, ckpt_name=None):
		print "load checkpoints ..."
		checkpoint_dir = checkpoint_dir or self.model_dir
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = ckpt_name or os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print "Success to read {}".format(ckpt_name)
			return True, counter
		else:
			print "Failed to find a checkpoint"
			return False, 0

def event_reader(event_path, event_name=None, names=[]):
    # get the newest event file
    if event_name is None:
        fs = os.listdir(event_path)
        fs.sort(key=lambda fn:os.path.getmtime(os.path.join(event_path, fn)))
        event_name = fs[-1]
    print "load from event:", os.path.join(event_path, event_name)
    res = {}
    for n in names:
        res[n] = []
    for e in tf.train.summary_iterator(os.path.join(event_path, event_name)):
        for v in e.summary.value:
            for n in names:
                if n == v.tag:
                    res[n].append(float(v.simple_value))
    return res