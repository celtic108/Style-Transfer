# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Heavily influenced by 
#    *  https://github.com/hwalsuklee/tensorflow-style-transfer
#    *  https://github.com/tensorflow/lucid
#    *  https://distill.pub/2018/differentiable-parameterizations/
#    *  https://github.com/VinceMarron/style_transfer

import os
import time
import numpy as np
from random import randint
#import pandas as pd
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
import tensorflow as tf

import lucid_ops
import wass_style_ops

from image_ops import *
from network_utils import *
        

def main(model,
         content_weights, 
         style_weights,
         learning_rate,
         alpha, beta, 
         cycles=float("inf"), 
         training_time=float("inf"), 
         start_jittering = 0.,
         stop_jittering=float("inf"), 
         jitter_freq=50, 
         display_image_freq=300, 
         max_image_dim = 512,
         pre_calc_style_grams = 0, 
         content_targets = 0,
         random_initializer = False,
         use_wass = False):
    
    # Content weights - Dictionary with the tensor names as the keys and the weights as the values
    # Style weights - Same as content weights but for style
    # Alpha, Beta - Total weighting of the content vs style respectively
    # Cycles - how many iterations to perform on each image. Set to float("inf") to remove (must specify time limit instead)
    # Training time - time limit for optimization. Set to float("inf") to remove (must specify training time instead)
    # Stop jittering - Float between 0 and 1. Fraction of the total allowed cycles or training time to jitter the image during optimization. Set to 0 for no image jittering
    # Jitter frequency - Number of training cycles between image shifts
    # Display image frequency - Number of training cycles between displaying the result image
    # Maximum image dimension - Scale down the largest dimension of the content image to match this
    
    
    if model in ['Inception_V1', 'Inception_V3']:
        from tensorflow.contrib.slim.nets import inception as model_module
    elif model == 'VGG_19':
        from tensorflow.contrib.slim.nets import vgg as model_module

    if (cycles == float("inf")) and (training_time == float("inf")):
        print("Error: Must specify time or cycle limit")
        return False
    jitter_stop_cycle = float("inf")
    jitter_stop_time = float("inf")
    jitter_start_cycle = 0
    jitter_start_time = 0
    if cycles < float("inf"):
        jitter_start_cycle = start_jittering * cycles
        if stop_jittering < float("inf"):
            jitter_stop_cycle = stop_jittering * cycles
    if training_time < float("inf"):
        jitter_start_time = start_jittering * training_time
        if stop_jittering < float("inf"):
            jitter_stop_time = stop_jittering * training_time

    slim = tf.contrib.slim
    
    content_image = load_images("./contentpicture", max_image_dim)
    print("Content Image: ", content_image.shape)
        
    style_image = load_images("./stylepicture", target_shape = content_image.shape)
    print("Style Image: ", style_image.shape)

    g=tf.Graph()
    with g.as_default():
        # Prepare graph
        var_input = tf.placeholder(shape = (None, content_image.shape[1], content_image.shape[2], 3), dtype=tf.float32, name='var_input')
        batch, h, w, ch = content_image.shape
        init_val, scale, corr_matrix = lucid_ops.get_fourier_features(content_image)

        decorrelate_matrix = tf.constant(corr_matrix, shape=[3,3])
        var_decor = tf.reshape(tf.matmul(tf.reshape(var_input, [-1, 3]), tf.matrix_inverse(decorrelate_matrix)), [1,content_image.shape[1],content_image.shape[2],3]) * 4.0
        four_space_complex = tf.spectral.rfft2d(tf.transpose(var_decor, perm=[0,3,1,2]))
        four_space_complex = four_space_complex / scale
        four_space = tf.concat([tf.real(four_space_complex), tf.imag(four_space_complex)], axis=0)
        
        four_input = tf.Variable(init_val)
        four_to_complex = tf.complex(four_input[0], four_input[1])
        four_to_complex = scale * four_to_complex

        rgb_space = tf.expand_dims(tf.transpose(tf.spectral.irfft2d(four_to_complex), perm = [1,2,0]), axis=0)
        rgb_space = rgb_space[:,:h,:w,:ch] / 4.0
        recorr_img = tf.reshape(tf.matmul(tf.reshape(rgb_space, [-1,3]), decorrelate_matrix), [1,content_image.shape[1],content_image.shape[2],3])
        input_img = (recorr_img + 1.0) / 2.0
        VGG_MEANS = np.array([[[[0.485, 0.456, 0.406]]]]).astype('float32')
        VGG_MEANS = tf.constant(VGG_MEANS, shape=[1,1,1,3])
        vgg_input = (input_img - VGG_MEANS) * 255.0
        bgr_input = tf.stack([vgg_input[:,:,:,2], 
                              vgg_input[:,:,:,1], 
                              vgg_input[:,:,:,0]], axis=-1)
        
        with g.gradient_override_map({'Relu': 'Custom1', 
                                      'Relu6': 'Custom2'}):
            if model == 'Inception_V1':
                with slim.arg_scope(model_module.inception_v1_arg_scope()):
                    _, end_points = model_module.inception_v1(
                    input_img, num_classes=1001, spatial_squeeze = False, is_training=False)
            elif model == 'Inception_V3':
                with slim.arg_scope(model_module.inception_v3_arg_scope()):
                    _, end_points = model_module.inception_v3(
                    input_img, num_classes=1001, spatial_squeeze = False, is_training=False)
            elif model == 'VGG_19':
                with slim.arg_scope(model_module.vgg_arg_scope()):
                    _, end_points = model_module.vgg_19(
                    bgr_input, num_classes=1000, spatial_squeeze = False, is_training=False)
        
        content_placeholders = {}
        content_losses = {}
        total_content_loss = 0
        style_losses = {}
        total_style_loss = 0
        input_grams = {}
        style_gram_placeholders = {}
        mean_placeholders = {}
        tr_cov_placeholders = {}
        root_cov_placeholders = {}
        means = {}
        tr_covs = {}
        root_covs = {}
        
        for layer in content_weights.keys():
            # Creates the placeholder for importing the content targets and creates the operations to compute the loss at each content layer
            _, h, w, d = g.get_tensor_by_name(layer).get_shape()
            
            content_placeholders[layer] = tf.placeholder(tf.float32, shape=[None,h,w,d])
            content_losses[layer] = tf.reduce_mean(tf.abs(content_placeholders[layer] - g.get_tensor_by_name(layer)))
            total_content_loss += content_losses[layer]*content_weights[layer]
            
        for layer in style_weights.keys():
            # Creates the placeholder for importing the pre-calculated style grams and creates the operations to compute the loss at each style layer
            _, h, w, d = g.get_tensor_by_name(layer).get_shape()
            N = h.value*w.value
            M = d.value
            if use_wass:
                means[layer], cov = wass_style_ops.calc_2_moments(g.get_tensor_by_name(layer))
                eigvals, eigvects = tf.self_adjoint_eig(cov)
                eigroot_mat = tf.diag(tf.sqrt(tf.maximum(eigvals, 0)))
                root_covs[layer] = tf.matmul(tf.matmul(eigvects, eigroot_mat), eigvects, transpose_b=True)
                tr_covs[layer] = tf.reduce_sum(tf.maximum(eigvals, 0))
                mean_placeholders[layer] = tf.placeholder(tf.float32, shape = means[layer].get_shape())
                tr_cov_placeholders[layer] = tf.placeholder(tf.float32, shape = tr_covs[layer].get_shape())
                root_cov_placeholders[layer] = tf.placeholder(tf.float32, shape = root_covs[layer].get_shape())
                style_losses[layer] = wass_style_ops.calc_l2wass_dist(mean_placeholders[layer], tr_cov_placeholders[layer], root_cov_placeholders[layer], means[layer], cov)
            else:
                input_grams[layer] = gram(g.get_tensor_by_name(layer))
                style_gram_placeholders[layer] = tf.placeholder(tf.float32, shape = input_grams[layer].get_shape())
                style_losses[layer] = tf.reduce_mean(tf.abs(input_grams[layer] - style_gram_placeholders[layer]))
            total_style_loss += style_weights[layer] * style_losses[layer]
	    
        total_loss = alpha*total_content_loss + beta*total_style_loss 
        
        update = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, var_list = [four_input])
        
        saver = tf.train.Saver(slim.get_model_variables())
            
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            restore_model(saver, model, sess)
            
            if display_image_freq < float("inf"):
                display_image(content_image)
                display_image(style_image)
            
            style_four_trans = sess.run(four_space, feed_dict = {var_input: preprocess(style_image)})
            copy_style_four_to_input_op = four_input.assign(style_four_trans)
            sess.run(copy_style_four_to_input_op)
            
            # Calculates the style grams for each style layer and saves them to feed to their placeholders
            pre_calc_style_grams = {}
            pre_calc_mean_placeholders = {}
            pre_calc_tr_cov_placeholders = {}
            pre_calc_root_cov_placeholders = {}
            for layer in style_weights.keys():
                print(layer)
                if use_wass:
                    pre_calc_mean_placeholders[layer] = sess.run(means[layer])
                    pre_calc_tr_cov_placeholders[layer] = sess.run(tr_covs[layer])
                    pre_calc_root_cov_placeholders[layer] = sess.run(root_covs[layer])
                else:
                    pre_calc_style_grams[layer] = sess.run(input_grams[layer])
            
            content_four_trans = sess.run(four_space, feed_dict = {var_input:preprocess(content_image)})
            copy_content_four_to_input_op = four_input.assign(content_four_trans)
            sess.run(copy_content_four_to_input_op)

            # Allows content targets to be used if they have already been calculated from a previous iteration
            content_targets = {}
            for layer in content_weights.keys():
                print(layer)
                content_targets[layer] = sess.run(g.get_tensor_by_name(layer))
            
            if random_initializer:
                reassign_random = four_input.assign(np.random.normal(size = (2, 3, content_image.shape[1], (content_image.shape[2] + 2) // 2),
                                                                     scale = 0.01))
                sess.run(reassign_random)

            assign_jitter = four_input.assign(four_space)
            
            # Generates the feed dictionary for session update           
            feed_dict = {}
            for layer in content_weights.keys():
                feed_dict[content_placeholders[layer]] = content_targets[layer]
            for layer in style_weights.keys():
                if use_wass:
                    feed_dict[mean_placeholders[layer]] = pre_calc_mean_placeholders[layer]
                    feed_dict[tr_cov_placeholders[layer]] = pre_calc_tr_cov_placeholders[layer]
                    feed_dict[root_cov_placeholders[layer]] = pre_calc_root_cov_placeholders[layer]
                else:
                    feed_dict[style_gram_placeholders[layer]] = pre_calc_style_grams[layer]
            start_time = time.time()
            i=0
            _, h, w, d = content_image.shape

            while ((i < cycles) and (time.time()-start_time < training_time)):
                # Perform update step
                loss, _, temp_image, tcl, tsl = sess.run([total_loss, update, recorr_img, total_content_loss, total_style_loss], feed_dict=feed_dict)
                if (i%jitter_freq==0 and i<jitter_stop_cycle and (i>jitter_start_cycle or time.time()-start_time > jitter_start_time)
                    and time.time()-start_time < jitter_stop_time):
                    temp_image = np.roll(temp_image, shift = randint(-1,1), axis = randint(1,2))
                    sess.run(assign_jitter, feed_dict={var_input:temp_image})
                # Print loss updates every 10 iterations
                if (i%10==0):
                    print(loss, i, tsl, tcl)
                # Display image 
                if display_image_freq < float("inf"):
                    if i%display_image_freq==0:
                        #image_out = un_preprocess(np.clip(sess.run(recorr_img), -1., 1.))
                        display_image(un_preprocess(np.clip(temp_image, -1., 1.)), True, i)
                i += 1
            
            # Display the final image and save it to the folder    
            image_out = un_preprocess(sess.run(recorr_img))
            display_image(image_out, save=True, name='final')
            if i>= cycles:
                print("Reached Cycle Limit: ", cycles)
            if (time.time()-start_time > training_time):
                print("Reached Time Limit: ", time.time()-start_time)
                
    
          
