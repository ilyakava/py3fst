import numpy as np
import tensorflow as tf
import glob
from util.tool import *
from networks.spectrogram_networks import Guo_Li_net
import time

audio_length = 19680
initial_bound = 0.1 # initial l infinity norm for adversarial perturbation
h = 257  # shape of the spectrogram (batch_size, h, w)
w = 31

class Attack:
    def __init__(self, sess, specs, labels, batch_size=1, lr_stage1=0.05, num_iter_stage1 = 1000):
        self.specs = specs
        self.labels = labels
        self.sess = sess 
        self.num_iter_stage1 = num_iter_stage1
        self.batch_size = batch_size     
        self.lr_stage1 = lr_stage1
        
        tf.set_random_seed(0)
        
        # placeholders
        self.spec_input =  tf.placeholder(tf.float32, shape=[batch_size, h, w], name="input_spec")
        self.target_ph  = tf.placeholder(tf.float32, shape=[batch_size, 3], name='target_labels')
        # variable
        self.delta = tf.Variable(np.zeros((batch_size, h, w), dtype=np.float32), name='delta')
        self.rescale = tf.Variable(np.ones((batch_size, 1, 1), dtype=np.float32), name='rescale')     
        # extract the delta             
        self.apply_delta = tf.clip_by_value(self.delta, -initial_bound, initial_bound) * self.rescale   
        self.new_inputs  = self.apply_delta + self.spec_input
        self.inputs = tf.clip_by_value(self.new_inputs, -2**15, 2**15-1)         
        # pass in to the network
        input_dict = {"spectrograms":self.inputs}
        self.output_ph = Guo_Li_net(input_dict, dropout=0.2, reuse=tf.AUTO_REUSE, 
                              is_training=False, n_classes=3, spec_h=h, spec_w=w)
        self.output_ph = tf.reshape(self.output_ph, [-1,3])
        self.loss   = tf.nn.softmax_cross_entropy_with_logits(labels=self.target_ph, logits=self.output_ph)
        self.total_loss = tf.reduce_mean(self.loss)
        
        self.optimizer1 = tf.train.AdamOptimizer(self.lr_stage1)
        self.train1 = self.optimizer1.minimize(self.total_loss, var_list=[self.delta])        
        
    def attack_stage1(self, target, verbose=0):
        
        sess = self.sess       
        # warm_start
        warm_start_from, id_assignment_map = warm_start()
        # initialize and load the pretrained model
        tf.train.init_from_checkpoint(warm_start_from, id_assignment_map)
        sess.run(tf.initialize_all_variables())
            
        # reassign the variables  
        sess.run(tf.assign(self.rescale, np.ones((self.batch_size, 1, 1), dtype=np.float32)))             
        sess.run(tf.assign(self.delta, np.zeros((self.batch_size, h, w), dtype=np.float32)))
        
        feed_dict = {self.spec_input: self.specs, self.target_ph: target}
     
        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iter_stage1
        final_deltas = [None] * self.batch_size
        clock = 0
        
        for i in range(MAX):           
            now = time.time()
            # Actually do the optimization                           
            sess.run(self.train1, feed_dict)
            delta, loss, predictions, new_inputs = sess.run((self.delta, self.loss, self.output_ph, self.new_inputs), feed_dict) 
            feed_dict = {self.spec_input: self.specs , self.target_ph: target}
            
            if i % 10 == 0 and verbose==1:
                print("Total adversarial loss at iteration {}:{}".format(i, np.mean(loss)))
                print("Perturbation sucess rate:{}".format(accuracy(predictions, target)))
                print("\n")
            # Sample five examples from the batch 
                if (self.batch_size >=5):
                    sampled_input = np.random.choice(np.arange(self.batch_size), (5,), replace=False)
                else:
                    sampled_input = np.random.choice(np.arange(self.batch_size), (1,), replace=False)
            for ii in range(self.batch_size):
                if (ii in sampled_input and verbose==1):
                    print("example: {}, loss: {}".format(ii, loss[ii]))
                    print("pred:{}".format(np.argmax(predictions[ii])))
                    print("target:{}".format(np.argmax(target[ii])))
                    print("true: {}".format(self.labels[ii]))
                    print("--------------------------------------------")
                if i % 100 == 0:
                    if np.argmax(predictions[ii]) == np.argmax(target[ii]):
                        # update rescale
                        rescale = sess.run(self.rescale)
                        if rescale[ii] * initial_bound > np.max(np.abs(delta[ii])):                            
                            rescale[ii] = np.max(np.abs(delta[ii])) / initial_bound                     
                        rescale[ii] *= .8
                        # save the best adversarial example
                        final_deltas[ii] = new_inputs[ii]                    
                        sess.run(tf.assign(self.rescale, rescale))
                                                      
                # in case no final_delta return        
                if (i == MAX-1 and final_deltas[ii] is None):
                    final_deltas[ii] = new_inputs[ii]             
         
            if i % 10 == 0:
                rescale = sess.run(self.rescale)
                print("mean rescale:{}".format(np.mean(rescale)))
                print("ten iterations take around {} ".format(clock))
                clock = 0
             
            clock += time.time() - now
            
        return np.array(final_deltas)