import numpy as np
import tensorflow as tf
import glob
from util.tool import *
from networks.spectrogram_networks import Guo_Li_net
import time
import os

audio_length = 19680
initial_bound = 0.08 # initial l infinity norm for adversarial perturbation
h = 257  # shape of the spectrogram (batch_size, h, w)
w = 31

class Attack:
    def __init__(self, sess, specs, labels, batch_size=1, lr_stage1=0.05, lr_stage2=0.5, num_iter_stage1 = 1000,
                num_iter_stage2 = 5000):
        self.specs = normalize(specs)
        self.labels = labels
        self.sess = sess 
        self.num_iter_stage1 = num_iter_stage1
        self.num_iter_stage2 = num_iter_stage2
        self.batch_size = batch_size     
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        tf.set_random_seed(0)
        
        # placeholders
        self.spec_input =  tf.placeholder(tf.float32, shape=[batch_size, h, w], name="input_spec")
        self.target_ph  = tf.placeholder(tf.float32, shape=[batch_size, 3], name='target_labels')
        self.psd_max_ori = tf.placeholder(tf.float32, shape=[batch_size], name='psd')
        self.th = tf.placeholder(tf.float32, shape=[batch_size, None, None], name='masking_threshold')
        
        # variable
        self.delta = tf.Variable(np.zeros((batch_size, h, w), dtype=np.float32), name='delta')
        self.rescale = tf.Variable(np.ones((batch_size, 1, 1), dtype=np.float32), name='rescale')
        self.alpha = tf.Variable(np.ones((batch_size), dtype=np.float32) * 0.000001, name='alpha')
        
        # add perturbation
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
        
        # compute the loss for masking threshold
        self.loss_th_list = []
        self.transform = Transform(window_size=400)
        for i in range(self.batch_size):
            logits_delta = self.transform((self.apply_delta[i, :]), (self.psd_max_ori)[i])
            loss_th =  tf.reduce_mean(tf.nn.relu(logits_delta - (self.th)[i]))            
            loss_th = tf.expand_dims(loss_th, dim=0) 
            self.loss_th_list.append(loss_th)
        self.loss_th = tf.concat(self.loss_th_list, axis=0)
        
        self.optimizer1 = tf.train.AdamOptimizer(self.lr_stage1)
        self.train1 = self.optimizer1.minimize(self.total_loss, var_list=[self.delta])        
        
        self.optimizer2 = tf.train.AdamOptimizer(self.lr_stage2)
        self.train21 = self.optimizer2.minimize(self.total_loss,var_list=[self.delta])
        self.train22 = self.optimizer2.minimize(self.alpha * self.loss_th, var_list=[self.delta])
        self.train2 = tf.group(self.train21, self.train22)       
        
    def attack_stage1(self, target, verbose=0):
        self.target = target
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
        count = 0
        
        for i in range(MAX):           
            now = time.time()
            # Actually do the optimization                           
            sess.run(self.train1, feed_dict)
            delta, loss, predictions, new_inputs = sess.run((self.delta, self.loss, self.output_ph, self.new_inputs), feed_dict) 
            feed_dict = {self.spec_input: self.specs , self.target_ph: target}
            
            sampled_input = []
            if i % 10 == 0 and verbose==1:
                print("Total adversarial loss at iteration {}:{}".format(i, np.mean(loss)))
                #print("Perturbation sucess rate:{}".format(accuracy(predictions, target)))
                print("Perturbation sucess rate:{}".format(count/self.batch_size))
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
                        if final_deltas[ii] is None:
                            count+=1
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
    
    def attack_stage2(self, adv, th_batch, psd_max_batch, verbose=0):
        sess = self.sess       
        # warm_start
        warm_start_from, id_assignment_map = warm_start()
        # initialize and load the pretrained model
        tf.train.init_from_checkpoint(warm_start_from, id_assignment_map)
        sess.run(tf.initialize_all_variables())
        
        sess.run(tf.assign(self.rescale, np.ones((self.batch_size, 1, 1), dtype=np.float32)))
        #sess.run(tf.assign(self.alpha, np.ones((self.batch_size), dtype=np.float32) * 0.0))
        sess.run(tf.assign(self.alpha, np.ones((self.batch_size), dtype=np.float64) * 1e-10))
        
        # reassign the perturbation
        sess.run(tf.assign(self.delta, adv))        
        
        feed_dict = {self.spec_input: self.specs, self.target_ph: self.target, self.psd_max_ori: psd_max_batch, self.th: th_batch}
        predictions, loss = sess.run((self.output_ph, tf.reduce_mean(self.loss_th)), feed_dict) 
        print("Perturbation sucess rate:{}".format(accuracy(predictions, self. target)))
        print("Original perceptual loss:{}".format(loss))
        print("\n")
       
        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iter_stage2
        loss_th = [np.inf] * self.batch_size
        final_deltas = list(adv)
        mask = [None] * self.batch_size
        final_alpha = [None] * self.batch_size
        
        clock = 0
        min_th = 0.0005 
        count = 0
        for i in range(MAX):           
            now = time.time()
            # Do the optimization                          
            sess.run(self.train2, feed_dict) 
            
            if i % 10 == 0:
                delta, loss, p_loss, predictions, new_inputs = sess.run((self.delta, self.loss, self.loss_th, self.output_ph, 
                                                                          self.new_inputs), feed_dict)
                if verbose == 1:
                    print("Total adversarial loss at iteration {}:{}".format(i, np.mean(loss)))
                    print("Total perceptual loss at iteration {}:{}".format(i, np.mean(p_loss)))
                    #print("Perturbation sucess rate:{}".format(accuracy(predictions, self.target)))
                    print("Perturbation sucess rate:{}".format(count/self.batch_size))
                    print("\n")
                    # Sample five examples from the batch 
                    if (self.batch_size >=5):
                        sampled_input = np.random.choice(np.arange(self.batch_size), (5,), replace=False)
                    else:
                        sampled_input = np.random.choice(np.arange(self.batch_size), (1,), replace=False)
                
            for ii in range(self.batch_size): 
                if i % 10 == 0:                                              
                    alpha = sess.run(self.alpha)                    
                    if (i % 100 == 0 and ii in sampled_input and verbose==1):
                        print("example: {}, loss: {}, perceptual loss: {}".format(ii, loss[ii], p_loss[ii]))
                        print("pred:{}".format(np.argmax(predictions[ii])))
                        print("target:{}".format(np.argmax(self.target[ii])))
                        print("true: {}".format(self.labels[ii]))
                        print("--------------------------------------------")
                        
                    # if the network makes the targeted prediction
                    if np.argmax(predictions[ii]) == np.argmax(self.target[ii]):
                        if p_loss[ii] < loss_th[ii]:
                            if (mask[ii] is None):
                                count+=1
                                mask[ii] = 0
                            final_deltas[ii] = new_inputs[ii]
                            loss_th[ii] = p_loss[ii] 
                            final_alpha[ii] = alpha[ii]
                           
                        # increase the alpha each 20 iterations    
                        if i % 10 == 0:                                
                            alpha[ii] *= 2.0
                            sess.run(tf.assign(self.alpha, alpha))
                                                
                    # if the network fails to make the targeted prediction, reduce alpha each 50 iterations
                    if i % 10 == 0 and np.argmax(predictions[ii]) != np.argmax(self.target[ii]):
                        alpha[ii] *= 0.8
                        alpha[ii] = max(alpha[ii], min_th)
                        sess.run(tf.assign(self.alpha, alpha))
           
                # in case no final_delta return        
                if (i == MAX-1 and final_deltas[ii] is None):
                    final_deltas[ii] = new_inputs[ii] 
            if i % 40 == 0:
                print("alpha is {}, loss_th is {}".format(final_alpha, loss_th))
            if i % 10 == 0:
                print("ten iterations take around {} ".format(clock))
                clock = 0
            if i % 100 == 0:
                print("Finish {}%".format(i*100/self.num_iter_stage2))
             
            clock += time.time() - now
        final_deltas = np.array(final_deltas)
        return final_deltas, loss_th, final_alpha
    
def main():
    _, batched_input, labels, th_batch, psd_max_batch = load_data()
    # Set the attack target
    target = np.zeros((328,3))
    target[:,0] = np.ones(328)
    batch_size=batched_input.shape[0]
    attack = Attack(tf.Session(), batched_input, labels, 
                    batch_size=batched_input.shape[0], 
                    lr_stage1=0.03, lr_stage2=0.05, 
                    num_iter_stage1 = 500, num_iter_stage2=50)
    adv = np.zeros(batched_input.shape)
    if (not os.path.isfile('pgd adversarial examples.npy')):
        print("----------------Attack Stage 1----------------------")
        adv_example = attack.attack_stage1(target, verbose=1)
        adv = adv_example - normalize(batched_input)
        with open('pgd adversarial examples.npy', 'wb') as f:
            np.save(f, unnormalize(adv_example))
    else:
        adv = normalize(np.load("pgd adversarial examples.npy")) - normalize(batched_input)
        attack.target = target
    print("----------------Attack Stage 2----------------------")
    adv_example, loss_th, final_alpha = attack.attack_stage2(adv, th_batch, psd_max_batch, verbose=1)
    with open('perceptual adversarial examples.npy', 'wb') as f:
        np.save(f, unnormalize(adv_example))
        
if __name__ == '__main__':
    main()   