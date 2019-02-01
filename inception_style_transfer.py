# Heavily influenced by https://github.com/hwalsuklee/tensorflow-style-transfer

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
from tensorflow.contrib.slim.nets import inception as inception

import lucid_ops



def display_image(gen_pixels, save=True, name=0):
    # Uses PIL Image to display images periodically during optimization
    if gen_pixels.shape[0]==1:
        height = gen_pixels.shape[1]
        width = gen_pixels.shape[2]
    else:
        height = gen_pixels.shape[0]
        width = gen_pixels.shape[1]
    print(gen_pixels.shape)
    #gen_pixels = gen_pixels.flatten()
    # Crops the 20 pixel border from the image
    # Ensure the image type matches the way the images were loaded
    #pix = Image.new('RGB', (width-2*20, height-2*20))
    #gen_image = pix.load()
    # Look for a more compact way to reorganize the pixel values
    #for x in range(width-2*20):
    #    for y in range(height-2*20):
    #        gen_image[x,y]=(int(gen_pixels[3*((y+20)*width+x+20)]),
    #        int(gen_pixels[3*((y+20)*width+x+20)+1]), 
    #        int(gen_pixels[3*((y+20)*width+x+20)+2]))
    #pix = pix.resize((width, height), Image.ANTIALIAS)
    pix = Image.fromarray(np.clip(gen_pixels[0],0,255).astype('uint8'))
    pix.show()
    if save:
        if not os.path.isdir('./trial_runs/'):
            os.makedirs('./trial_runs/')
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

#def add_one_dim(image):
#    shape = (1,) + image.shape
#    return np.reshape(image, shape)

def gram(a):
    # Creates the TF operations to calculate the style gram of an image at a given level
    shape = a.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(a, shape=[-1,num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    length = tf.shape(matrix)[0]
    gram /= tf.cast(length, tf.float32)
    return gram

#def add_border(image,width):
#    temp = np.zeros((image.shape[0]+2*width,image.shape[1]+2*width,3))
#    for i in range(image.shape[0]):
#        for j in range(image.shape[1]):
#            for k in range(3):
#                temp[i+width,j+width,k] = image[i,j,k]
#    return temp

#def remove_border(image,width):
#    temp = np.zeros((image.shape[0]-2*width,image.shape[1]-2*width,3))
#    for i in range(image.shape[0]):
#        for j in range(image.shape[1]):
#            for k in range(3):
#                temp[i,j,k] = image[i+width,j+width,k]
#    return temp

"""def load_images(input_dir, batch_shape, max_image_dim = None, reshape=False, dims=0):
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
            #image = add_border(image,20)
        elif (image.shape[1] > max_image_dim) and (image.shape[1] > image.shape[0]):
            image = imresize(image,(max_image_dim*image.shape[0]//image.shape[1], max_image_dim,3), interp='nearest')
            #image = add_border(image,20)
        
        images.append(image)
        idx += 1
        if idx == batch_size:
            yield np.asarray(images)
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield np.asarray(images)
"""

def load_images(input_dir, max_image_dim = float('inf'), target_shape = None):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            try:
                image = Image.open(os.path.join(root, file)).convert("RGB")
                w, h = image.size
                if w > h:
                    if h > max_image_dim:
                        image = image.resize((max_image_dim, max_image_dim), resample = Image.LANCZOS)
                    elif w > max_image_dim:
                        image = image.resize((max_image_dim, h), resample = Image.LANCZOS)
                else:
                    if w > max_image_dim:
                        image = image.resize((max_image_dim, max_image_dim), resample = Image.LANCZOS)
                    elif w > max_image_dim:
                        image = image.resize((w, max_image_dim), resample = Image.LANCZOS)
                if target_shape is not None:
                    print("Are we trying this?")
                    w_diff = max(0, w-target_shape[1])
                    h_diff = max(0, h-target_shape[2])
                    image = image.resize((max(target_shape[1], w), max(target_shape[2], h)), resample = Image.LANCZOS)
                    offset_x = randint(0,w_diff)
                    offset_y = randint(0,h_diff)
                    print("New Size: ", offset_x, offset_y, target_shape[1]+ offset_x, target_shape[2]+offset_y)
                    image = image.crop((offset_x, offset_y, target_shape[1] + offset_x, target_shape[2] + offset_y))
                return np.expand_dims(np.array(image), axis=0)
            except:
                print("Found file that was not an image. Keep looking.")


def calc_2_moments(tensor):
  """flattens tensor and calculates sample mean and covariance matrix 
  along last dim (presumably channels)"""
  
  shape = tf.shape(tensor, out_type=tf.int32)
  n = tf.reduce_prod(shape[:-1])
  
  flat_array = tf.reshape(tensor, (n, shape[-1]))
  mu = tf.reduce_mean(flat_array, axis=0, keepdims=True)
  cov = (tf.matmul(flat_array - mu,flat_array - mu, transpose_a=True)/
                    tf.cast(n, tf.float32))
  
  return mu, cov

def calc_l2wass_dist(mean_stl, tr_cov_stl, root_cov_stl, mean_synth, cov_synth):
  """Calculates (squared) l2-Wasserstein distance between gaussians
  parameterized by first two moments of style and synth activations"""
  
  #tr_cov_synth = tf.trace(cov_synth)
  tr_cov_synth = tf.reduce_sum(tf.maximum(
                tf.self_adjoint_eig(cov_synth)[0],0.))
  
  
  mean_diff_squared = tf.reduce_sum(tf.square(mean_stl-mean_synth))

  cov_prod = tf.matmul(tf.matmul(root_cov_stl,cov_synth),root_cov_stl)
  
  #trace of sqrt of matrix is sum of sqrts of eigenvalues
  var_overlap = tf.reduce_sum(tf.sqrt(tf.maximum(
                tf.self_adjoint_eig(cov_prod)[0],0.1)))

  #loss can be slightly negative because of the 'maximum' on eigvals of cov_prod
  #could fix with  tr_cov_synth= tf.reduce_sum(tf.maximum(cov_synth,0))
  #but that would mean extra non-critical computation

  dist = mean_diff_squared+tr_cov_stl+tr_cov_synth-2*var_overlap
  
  ### above dist written out in latec:
  #\mathcal{W}_2(\mathcal{N}(\mu_{x},\Sigma_{x}),\mathcal{N}(\mu_{y},\Sigma_{y}))^2
  #&= \inf_{g \in G(\mathcal{N}^x,\mathcal{N}^y)} \mathbb{E}_{g}||x-y||^2 \\
  #&= ||\mu_x-\mu_y||^2 + \mbox{tr} (\Sigma_x)+ \mbox{tr} (\Sigma_y) 
  #- 2\mbox{tr} \left((\Sigma_y^{\frac{1}{2}}\Sigma_x\Sigma_y^{\frac{1}{2}})^{\frac{1}{2}}\right)
  
  
  return dist
                      

def rotate_image(image, angle):
    image = un_preprocess(image)
    height, width, _ = image.shape
    # Improvement - Find an image rotation algorithm that doesn't leave behind artifacts
    image = rotate(image, angle, reshape=False, mode='wrap')
    return preprocess(image)

def preprocess(image):
    #image = ((image/255.)-0.5)*2
    image = image/255.
    return image
    
def un_preprocess(image):
    return 255.*image
    #return 255.*(image +1)/2.

def main(model,
         content_weights, 
         style_weights,
         learning_rate,
         alpha, beta, 
         cycles=float("inf"), 
         training_time=float("inf"), 
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
    if model == 'Inception_V1':
        num_classes = 1001
    else:
        num_classes = 1000
    
    content_image = load_images("./contentpicture", max_image_dim)
    print("Content IMage: ", content_image.shape)
        
    style_image = load_images("./stylepicture", target_shape = content_image.shape)
    print("Style IMage: ", style_image.shape)

    g=tf.Graph()
    with g.as_default():
        # Prepare graph
        #var_input = tf.Variable(tf.random_uniform([1,content_image.shape[1],content_image.shape[2],3], minval=-1, maxval=1), name="CombinedImage", dtype=tf.float32)
        var_input = tf.placeholder(shape = (None, content_image.shape[1], content_image.shape[2], 3), dtype=tf.float32, name='var_input')
        batch, h, w, ch = content_image.shape
        init_val, scale, corr_matrix = lucid_ops.get_fourier_features(content_image)

        #corr_matrix = ccss
        decorrelate_matrix = tf.constant(corr_matrix, shape=[3,3])
        #un_sigmoid = -1*tf.log(tf.pow(var_input, -1)-1)
        #VGG_MEANS = np.array([[[[123.68, 116.78, 103.94]]]]).astype('float32')
        #VGG_MEANS = tf.constant(VGG_MEANS, shape=[1,1,1,3])
        var_decor = tf.reshape(tf.matmul(tf.reshape(var_input, [-1, 3]), tf.matrix_inverse(decorrelate_matrix)), [1,content_image.shape[1],content_image.shape[2],3]) * 4.0
        four_space_complex = tf.spectral.rfft2d(tf.transpose(var_decor, perm=[0,3,1,2]))
        #four_space_complex = tf.spectral.rfft2d(var_decor)
        four_space_complex = four_space_complex / scale
        four_space = tf.concat([tf.real(four_space_complex), tf.imag(four_space_complex)], axis=0)
        #init_val = np.random.randn(2, 3, content_image.shape[1], (content_image.shape[2]+2) // 2)
        
        four_input = tf.Variable(init_val)
        four_to_complex = tf.complex(four_input[0], four_input[1])
        four_to_complex = scale * four_to_complex

        
        #four_input = tf.Variable(init_val, dtype=tf.float32)
        #four_to_complex = tf.complex(four_input[0], four_input[1])
        rgb_space = tf.expand_dims(tf.transpose(tf.spectral.irfft2d(four_to_complex), perm = [1,2,0]), axis=0)
        rgb_space = rgb_space[:,:h,:w,:ch] / 4.0
        #rgb_space = tf.spectral.irfft2d(four_to_complex)
        recorr_img = tf.reshape(tf.matmul(tf.reshape(rgb_space, [-1,3]), decorrelate_matrix), [1,content_image.shape[1],content_image.shape[2],3])
        if model == 'Inception_V1':
            #input_img = (recorr_img - 0.5) * 2.0
            input_img = tf.nn.sigmoid(recorr_img)
        else:
            input_img = 255.*recorr_img + VGG_MEANS
        #pre_processed_image = preprocess(recorr_img_pre)
        #crops = []
        #for _ in range(7):
        #    crops.append(tf.random_crop(tf.pad(input_img, [[0,0],[8,8],[8,8],[0,0]]), [1, content_image.shape[1], content_image.shape[2], 3]))
        #input_img = tf.concat([input_img, *crops], axis = 0)

        with g.gradient_override_map({'Relu': 'Custom1', 
                                      'Relu6': 'Custom2'}):


            if model == 'Inception_V1':
                with slim.arg_scope(inception.inception_v1_arg_scope()):
                    _, end_points = inception.inception_v1(
                    recorr_img, num_classes=num_classes, spatial_squeeze = False, is_training=False)
            elif model == 'Inception_V3':
                with slim.arg_scope(inception.inception_v3_arg_scope()):
                    _, end_points = inception.inception_v3(
                    input_img, num_classes=num_classes, spatial_squeeze = False, is_training=False)
        
        
        #layer_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        #for laynam in layer_names:
        #    if '4' in laynam:
        #        print(laynam)
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
            if model == 'Inception_V1':
                content_losses[layer] = tf.reduce_mean(tf.abs(content_placeholders[layer] - g.get_tensor_by_name(layer)))
            #content_losses[layer] = (1./(2.*np.sqrt(w.value*h.value)*np.sqrt(d.value)))*tf.reduce_sum(tf.pow(content_placeholders[layer] - g.get_tensor_by_name(layer),2))
            #content_losses[layer] = tf.reduce_sum(tf.pow(content_placeholders[layer] - g.get_tensor_by_name(layer), 2))
            total_content_loss += content_losses[layer]*content_weights[layer]
            
        for layer in style_weights.keys():
            # Creates the placeholder for importing the pre-calculated style grams and creates the operations to compute the loss at each style layer
            _, h, w, d = g.get_tensor_by_name(layer).get_shape()
            N = h.value*w.value
            M = d.value
            if use_wass:
                means[layer], cov = calc_2_moments(g.get_tensor_by_name(layer))
                eigvals, eigvects = tf.self_adjoint_eig(cov)
                eigroot_mat = tf.diag(tf.sqrt(tf.maximum(eigvals, 0)))
                root_covs[layer] = tf.matmul(tf.matmul(eigvects, eigroot_mat), eigvects, transpose_b=True)
                tr_covs[layer] = tf.reduce_sum(tf.maximum(eigvals, 0))
                mean_placeholders[layer] = tf.placeholder(tf.float32, shape = means[layer].get_shape())
                tr_cov_placeholders[layer] = tf.placeholder(tf.float32, shape = tr_covs[layer].get_shape())
                root_cov_placeholders[layer] = tf.placeholder(tf.float32, shape = root_covs[layer].get_shape())
                style_losses[layer] = calc_l2wass_dist(mean_placeholders[layer], tr_cov_placeholders[layer], root_cov_placeholders[layer], means[layer], cov)
            else:
                input_grams[layer] = gram(g.get_tensor_by_name(layer))
                style_gram_placeholders[layer] = tf.placeholder(tf.float32, shape = input_grams[layer].get_shape())
                style_losses[layer] = style_weights[layer] * tf.reduce_mean(tf.abs(input_grams[layer] - style_gram_placeholders[layer]))
            #style_losses[layer] = style_weights[layer]*(1. / (4*N**2 * M**2)) * tf.reduce_sum(tf.pow(input_grams[layer] - style_gram_placeholders[layer], 2))
            #style_losses[layer] = style_weights[layer] * tf.reduce_sum(tf.pow(input_grams[layer] - style_gram_placeholders[layer], 2))
            total_style_loss += style_losses[layer]
	    
        total_loss = alpha*total_content_loss + beta*total_style_loss 
        
        # gradient and update function WITH clipping
        # I caluclated the gradient and then multiplied by inf and clipped to 1/255. This forces all pixels to change unless the gradient is exactly zero and also limits each 
        #    change to 1 increment at each iteration. The quality of the image is not great but it is not clear if this is because of the optimization method or the underlying network
        #    VGG-type networks seem to work much better in general. Try using this optimization method with VGGs
        
        
        if model == 'Inception_V1':
            #optimizer = tf.train.AdamOptimizer(0.05)
            #gradient = optimizer.compute_gradients(total_loss, var_list = [four_input])
            update = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, var_list = [four_input])
        
        #gradient = tf.reshape(tf.gradients(xs = four_input, ys = total_loss), four_input.get_shape())
        #update = four_input.assign(tf.add(four_input, -learning_rate*gradient))
        
        saver = tf.train.Saver(slim.get_model_variables())
            
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            #saver.restore(sess, './input/inception_v3.ckpt')
            #saver.restore(sess, './input/vgg_19.ckpt')
            try:
                saver.restore(sess, './input/inception_v1.ckpt')
            except:
                ans = input("Model not found. Would you like to download it?: [y/n]")
                if ans in ['Y', 'y', 'Yes', 'YES', 'yes']:
                    if not os.path.isdir('./input'):
                        os.makedirs('./input')
                    import urllib.request
                    import shutil
                    url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
                    file_name = './input/inception_v1_2016_08_28.tar.gz'
                    print("Downloading model")
                    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                    print("Extracting tar.gz")
                    import tarfile
                    tar = tarfile.open(file_name, 'r:gz')
                    tar.extractall(path='./input/')
                    tar.close()
                    os.remove(file_name)
                    
                    saver.restore(sess, './input/inception_v1.ckpt')

            #op_list = sess.graph.get_operations()
            #for op in op_list:
            #    print("OP: ", op.name)

            if display_image_freq < float("inf"):
                display_image(content_image)
                display_image(style_image)
            
            #style_assign_op = var_input.assign(preprocess(style_image))
            #sess.run(style_assign_op)
            #whitened_image = sess.run(var_decor)
            whitened_image = sess.run(var_decor, feed_dict={var_input:preprocess(style_image)})
            #display_image(un_preprocess(whitened_image))
            print(type(whitened_image))
            print(whitened_image.shape)
            flat_white = whitened_image.reshape((-1,3))


            #****
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt

            def plot_scatter(x):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
    
                n = 100
                idx = np.random.randint(x.shape[0], size=1001)
                # For each set of style and range settings, plot n random points in the box
                # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
                ax.scatter(x[idx,0], x[idx,1], x[idx,2], c='r', marker='o')
    
                ax.set_xlabel('Red')
                ax.set_ylabel('Green')
                ax.set_zlabel('Blue')
    
                plt.show()
                #****
            
            #plot_scatter(style_image.reshape((-1,3)))
            #plot_scatter(preprocess(style_image).reshape((-1,3)))
            #plot_scatter(flat_white)
            
            #style_four_trans = sess.run(four_space)
            style_four_trans = sess.run(four_space, feed_dict = {var_input: preprocess(style_image)})
            #print(style_four_trans.shape)
            #print("FOUR SCALE")
            copy_style_four_to_input_op = four_input.assign(style_four_trans)
            sess.run(copy_style_four_to_input_op)
            
            #in_img = sess.run(recorr_img)
            #plot_scatter(in_img.reshape((-1,3)))
            #display_image(in_img)
            
            #recreated_style = sess.run(recorr_img)
            #display_image(un_preprocess(recreated_style))
            #plot_scatter(recreated_style.reshape((-1,3)))
            #print("RECREATED", recreated_style)
            #print("ORIGINAL", style_image)
            
            # Allows style grams to be used if they have already been calculated from a previous iteration
            if pre_calc_style_grams == 0:
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
            
            #content_assign_op = var_input.assign(preprocess(content_image))
            #content_assign_op = var_input.assign(content_image)
            #sess.run(content_assign_op)
            #temp_image = sess.run(var_decor)
            #temp_image = sess.run(var_decor, feed_dict={var_input:preprocess(content_image)})
            #display_image(un_preprocess(temp_image))
            content_four_trans = sess.run(four_space, feed_dict = {var_input:preprocess(content_image)})
            copy_content_four_to_input_op = four_input.assign(content_four_trans)
            sess.run(copy_content_four_to_input_op)
            
            
            # Allows content targets to be used if they have already been calculated from a previous iteration
            if content_targets == 0:
                # Calculates the content targets to feed to their placeholders
                content_targets = {}
                for layer in content_weights.keys():
                    print(layer)
                    content_targets[layer] = sess.run(g.get_tensor_by_name(layer))
            
            #init_val = np.random.randn(2, 3, content_image.shape[1], (content_image.shape[2]+2) // 2)
            #new_init = four_input.assign(init_val)
            #temp_image = sess.run(new_init)
            
            
            #init_val = np.random.randn(1,content_image.shape[1], content_image.shape[2], 3)/100  + 0.5
            #init_val[0,0:20,0:20,:] = (1,-1,-1)
            #new_init = var_input.assign(init_val)
            #temp_image = sess.run(new_init)
            #temp_four = sess.run(four_space)
            #temp_four = sess.run(four_space, feed_dict={var_input:init_val})
            #init_four_trans = four_input.assign(temp_four)
            #if random_initializer:
            #    sess.run(init_four_trans)
            #new_input = sess.run(recorr_img)
            #plot_scatter(new_input.reshape((-1,3)))
     
            if random_initializer:
                init_val = sess.run(four_space, feed_dict = {var_input: np.random.rand(*content_image.shape)/100 + 0.5})
                reassign_random = four_input.assign(init_val)
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
                # Jitter image if applicable
                #if (i%jitter_freq==0 and i<jitter_stop_cycle and time.time()-start_time < jitter_stop_time):
                #    jitter_op = var_input.assign(np.reshape(jitter_image(np.clip(np.reshape(temp_image, [h,w,3]), a_min=-1, a_max=1),i//jitter_freq),[1,h,w,3]))
                #    temp_image = sess.run(jitter_op)
                if (i%jitter_freq==0 and i<jitter_stop_cycle and time.time()-start_time < jitter_stop_time):
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
                
    return (pre_calc_style_grams, content_targets)

          
