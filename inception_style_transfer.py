# Heavily influenced by https://github.com/hwalsuklee/tensorflow-style-transfer

import os
import time
import numpy as np
#import pandas as pd
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception


def display_image(gen_pixels, save=True, name=0):
    # Uses PIL Image to display images periodically during optimization
    if gen_pixels.shape[0]==1:
        height = gen_pixels.shape[1]
        width = gen_pixels.shape[2]
    else:
        height = gen_pixels.shape[0]
        width = gen_pixels.shape[1]
    gen_pixels = gen_pixels.flatten()
    # Crops the 20 pixel border from the image
    # Ensure the image type matches the way the images were loaded
    pix = Image.new('RGB', (width-2*20, height-2*20))
    gen_image = pix.load()
    # Look for a more compact way to reorganize the pixel values
    for x in range(width-2*20):
        for y in range(height-2*20):
            gen_image[x,y]=(int(gen_pixels[3*((y+20)*width+x+20)]),
            int(gen_pixels[3*((y+20)*width+x+20)+1]), 
            int(gen_pixels[3*((y+20)*width+x+20)+2]))
    pix = pix.resize((width, height), Image.ANTIALIAS)
    pix.show()
    if save:
        pix.save("./trial_runs/"+str(name)+".jpg")

def jitter_image(image, key=-1):
    # Either rotates the image 1 degree in either direction or shifts it 1 pixel in any direction
    # if no key is provided, a random shift is selected. Key allows for cyclical shifting
    if key == -1:
        key = np.random.randint(0,6)
    if key%6 ==0:
        return rotate_image(image, 1)
    if key%6 ==1:
        return np.roll(image,image.shape[1]-1,axis=1)
    if key%6 == 2:
        return np.roll(image,1,axis=1)
    if key%6 == 3:
        return rotate_image(image, -1)
    if key%6 == 4:
        return np.roll(image,1,axis=0)
    if key%6 == 5:
        return np.roll(image,image.shape[0]-1,axis=0)
    return image

def add_one_dim(image):
    shape = (1,) + image.shape
    return np.reshape(image, shape)

def gram(a):
	# Creates the TF operations to calculate the style gram of an image at a given level
	shape = a.get_shape()
	num_channels = int(shape[3])
	matrix = tf.reshape(a, shape=[-1,num_channels])
	gram = tf.matmul(tf.transpose(matrix), matrix)
	return gram

def add_border(image,width):
    temp = np.zeros((image.shape[0]+2*width,image.shape[1]+2*width,3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                temp[i+width,j+width,k] = image[i,j,k]
    return temp

#def remove_border(image,width):
#    temp = np.zeros((image.shape[0]-2*width,image.shape[1]-2*width,3))
#    for i in range(image.shape[0]):
#        for j in range(image.shape[1]):
#            for k in range(3):
#                temp[i,j,k] = image[i+width,j+width,k]
#    return temp

def load_images(input_dir, batch_shape, max_image_dim, reshape=False, dims=0):
    images = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.*'))[:20]:
        with tf.gfile.Open(filepath, "rb") as f:
            image = imread(f, mode='RGB').astype(np.float)
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        # Allows for the content image to be automatically resized to match the max dimension
        #   and reshape the style image to match the content image size (after border add)
        if reshape:
            image = imresize(image,(dims[1],dims[2],3), interp='nearest')
        elif (image.shape[0] > max_image_dim) and (image.shape[0] >= image.shape[1]):
            image = imresize(image,(max_image_dim,max_image_dim*image.shape[1]//image.shape[0],3), interp='nearest')
            image = add_border(image,20)
        elif (image.shape[1] > max_image_dim) and (image.shape[1] > image.shape[0]):
            image = imresize(image,(max_image_dim*image.shape[0]//image.shape[1], max_image_dim,3), interp='nearest')
            image = add_border(image,20)
        
        images.append(image)
        idx += 1
        if idx == batch_size:
            yield np.asarray(images)
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield np.asarray(images)
        
def rotate_image(image, angle):
    image = un_preprocess(image)
    height, width, _ = image.shape
    # Improvement - Find an image rotation algorithm that doesn't leave behind artifacts
    image = rotate(image, angle, reshape=False, mode='wrap')
    return preprocess(image)

def preprocess(image):
    return 2.*image/255. -1.
    
def un_preprocess(image):
    return 255.*(image +1.)/2.

def main(content_weights, 
         style_weights,
         alpha, beta, 
         cycles=float("inf"), 
         training_time=float("inf"), 
         stop_jittering=float("inf"), 
         jitter_freq=50, 
         display_image_freq=300, 
         max_image_dim = 512,
         pre_calc_style_grams = 0, 
         content_targets = 0):
    
    # Content weights - Dictionary with the tensor names as the keys and the weights as the values
    # Style weights - Same as content weights but for style
    # Alpha, Beta - Total weighting of the content vs style respectively
    # Cycles - how many iterations to perform on each image. Set to float("inf") to remove (must specify time limit instead)
    # Training time - time limit for optimization. Set to float("inf") to remove (must specify training time instead)
    # Stop jittering - Float between 0 and 1. Fraction of the total allowed cycles or training time to jitter the image during optimization. Set to 0 for no image jittering
    # Jitter frequency - Number of training cycles between image shifts
    # Display image frequency - Number of training cycles between displaying the result image
    # Maximum image dimension - Scale down the largest dimension of the content image to match this
    
    # Returns the pre-calculated style grams and content targets. This allows multiple trials without the need for recalculating these each time
    
    if (cycles == float("inf")) and (training_time == float("inf")):
        print("Error: Must specify time or cycle limit")
        return False
    jitter_stop_cycle = float("inf")
    jitter_stop_time = float("inf")
    if (cycles < float("inf") and stop_jittering < float("inf")):
        jitter_stop_cycle = stop_jittering * cycles
    if (training_time < float("inf") and stop_jittering < float("inf")):
        jitter_stop_time = stop_jittering * training_time
		
    # With line 165 "with slim.arg_scope..." - Investigate to find out how to import Inception V3 without any overhead.
    # Limit imported model to the highest level specified in the content and style weights
    slim = tf.contrib.slim
    num_classes = 1001
    
    for images in load_images("./contentpicture", [1], max_image_dim):
        # Normalized the content image instead of preprocessing because I was using a rather dark picture
        content_image = (images-np.average(images))/(2.*np.std(images))
        
    for images in load_images("./stylepicture", [1], max_image_dim, True, content_image.shape):
        style_image = preprocess(images)
    
    g=tf.Graph()
    with g.as_default():
        # Prepare graph
        var_input = tf.Variable(tf.random_uniform([1,content_image.shape[1],content_image.shape[2],3], minval=-1, maxval=1), name="CombinedImage", dtype=tf.float32)
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
            var_input, num_classes=num_classes, spatial_squeeze = False, is_training=False)
            
        content_placeholders = {}
        content_losses = {}
        total_content_loss = 0
        style_losses = {}
        total_style_loss = 0
        input_grams = {}
        style_gram_placeholders = {}
        
        for layer in content_weights.keys():
            # Creates the placeholder for importing the content targets and creates the operations to compute the loss at each content layer
            _, h, w, d = g.get_tensor_by_name(layer).get_shape()
            
            content_placeholders[layer] = tf.placeholder(tf.float32, shape=[1,h,w,d])
            content_losses[layer] = (1./(2.*np.sqrt(w.value*h.value)*np.sqrt(d.value)))*tf.reduce_sum(tf.pow(content_placeholders[layer] - g.get_tensor_by_name(layer),2))
            total_content_loss += content_losses[layer]*content_weights[layer]
            
        for layer in style_weights.keys():
            # Creates the placeholder for importing the pre-calculated style grams and creates the operations to compute the loss at each style layer
            _, h, w, d = g.get_tensor_by_name(layer).get_shape()
            N = h.value*w.value
            M = d.value
            input_grams[layer] = gram(g.get_tensor_by_name(layer))
            style_gram_placeholders[layer] = tf.placeholder(tf.float32, shape = input_grams[layer].get_shape())
            style_losses[layer] = style_weights[layer]*(1. / (4*N**2 * M**2)) * tf.reduce_sum(tf.pow(input_grams[layer] - style_gram_placeholders[layer], 2))
            total_style_loss += style_losses[layer]
	    
        total_loss = alpha*total_content_loss + beta*total_style_loss 
        
        # gradient and update function WITH clipping
        # I caluclated the gradient and then multiplied by inf and clipped to 1/255. This forces all pixels to change unless the gradient is exactly zero and also limits each 
        #    change to 1 increment at each iteration. The quality of the image is not great but it is not clear if this is because of the optimization method or the underlying network
        #    VGG-type networks seem to work much better in general. Try using this optimization method with VGGs
        gradient = float("inf")*tf.reshape(tf.gradients(xs = var_input, ys = total_loss), var_input.get_shape())
        update = var_input.assign(tf.clip_by_value(tf.add(var_input, -1*tf.clip_by_value(gradient, -1./255., 1./255.)), -1., 1.))
        
        saver = tf.train.Saver(slim.get_model_variables())
            
        with tf.Session() as sess:
            saver.restore(sess, './input/inception-v3/inception_v3.ckpt')
            
            if display_image_freq < float("inf"):
                display_image(un_preprocess(content_image))
                display_image(un_preprocess(style_image))
            
            style_assign_op = var_input.assign(style_image)
            sess.run(style_assign_op)
            
            # Allows style grams to be used if they have already been calculated from a previous iteration
            if pre_calc_style_grams == 0:
                # Calculates the style grams for each style layer and saves them to feed to their placeholders
                pre_calc_style_grams = {}
                for layer in style_weights.keys():
                    pre_calc_style_grams[layer] = sess.run(input_grams[layer])
            
            content_assign_op = var_input.assign(content_image)
            sess.run(content_assign_op)
            
            # Allows content targets to be used if they have already been calculated from a previous iteration
            if content_targets == 0:
                # Calculates the content targets to feed to their placeholders
                content_targets = {}
                for layer in content_weights.keys():
                    content_targets[layer] = sess.run(g.get_tensor_by_name(layer))
             
            # Generates the feed dictionary for session update           
            feed_dict = {}
            for layer in content_weights.keys():
                feed_dict[content_placeholders[layer]] = content_targets[layer]
            for layer in style_weights.keys():
                feed_dict[style_gram_placeholders[layer]] = pre_calc_style_grams[layer]
            start_time = time.time()
            i=0
            _, h, w, d = content_image.shape
            while ((i < cycles) and (time.time()-start_time < training_time)):
                # Perform update step
                loss, _, temp_image = sess.run([total_loss, update, var_input], feed_dict=feed_dict)
                # Jitter image if applicable
                if (i%jitter_freq==0 and i<jitter_stop_cycle and time.time()-start_time < jitter_stop_time):
                    jitter_op = var_input.assign(np.reshape(jitter_image(np.clip(np.reshape(temp_image, [h,w,3]), a_min=-1, a_max=1),i//jitter_freq),[1,h,w,3]))
                    temp_image = sess.run(jitter_op)
                # Print loss updates every 10 iterations
                if (i%10==0):
                    print(loss, i)
                # Display image 
                if display_image_freq < float("inf"):
                    if i%display_image_freq==0:
                        image_out = un_preprocess(sess.run(var_input))
                        display_image(image_out, True, i)
                i += 1
            
            # Display the final image and save it to the folder    
            image_out = un_preprocess(sess.run(var_input))
            display_image(image_out, save=True, name='final')
            if i>= cycles:
                print("Reached Cycle Limit: ", cycles)
            if (time.time()-start_time > training_time):
                print("Reached Time Limit: ", time.time()-start_time)
                
    return (pre_calc_style_grams, content_targets)

          
