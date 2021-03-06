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

import inception_style_transfer as istr
import numpy as np

model = 'Inception_V1'

content_weights = {}
style_weights = {}
hours = 7.0
minutes = 6.0
training_time = 60*(60*hours + minutes)
display_image_freq=100
jitter_freq = 3
cycles=1000
start_jittering = 0.1
stop_jittering= .75
max_image_dim = 768  
random_initializer = True


if model == 'Inception_V1':
    use_wass = False
    if use_wass:
        alpha = 800   #Content Weight
        beta = 1      #Style Weight
        display_image_freq = 10
        cycles = 200
    else:
        alpha = 1     #Content Weight
        beta = 2.5    #Style Weight
    learning_rate = .005               
    
    content_weights["InceptionV1/InceptionV1/Mixed_3b/concat:0"] = 1
    
    style_weights["InceptionV1/InceptionV1/Conv2d_2c_3x3/Relu:0"] = 1
    style_weights["InceptionV1/InceptionV1/MaxPool_3a_3x3/MaxPool:0"] = 1
    style_weights["InceptionV1/InceptionV1/MaxPool_4a_3x3/MaxPool:0"] = 1
    style_weights["InceptionV1/InceptionV1/Mixed_4b/concat:0"] = 1
    style_weights["InceptionV1/InceptionV1/Mixed_4c/concat:0"] = 1

elif model == 'Inception_V3':    
    use_wass = False
    if use_wass:
        alpha = 200   #Content Weight
        beta = 1      #Style Weight
        display_image_freq = 10
        cycles = 200
        learning_rate = 0.02
        style_weights["InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu:0"] = 1
        style_weights["InceptionV3/InceptionV3/MaxPool_3a_3x3/MaxPool:0"] = 1
    else:
        alpha = 1     #Content Weight
        beta = 1      #Style Weight
        learning_rate = .02         
    
    content_weights["InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu:0"] = 1
    
    style_weights["InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu:0"] = 1
    style_weights["InceptionV3/InceptionV3/MaxPool_5a_3x3/MaxPool:0"] = 1
    style_weights["InceptionV3/InceptionV3/Mixed_5b/concat:0"] = 1
    style_weights["InceptionV3/InceptionV3/Mixed_5c/concat:0"] = 1
    style_weights["InceptionV3/InceptionV3/Mixed_6a/concat:0"] = 0.5
    
elif model == 'VGG_19':    
    use_wass = True
    if use_wass:
        jitter_freq = float('inf')
        alpha = 1      #Content Weight
        beta = 5e-6    #Style Weight
        display_image_freq = 10
        cycles = 200
        learning_rate = 0.02
    else:
        alpha = 1     #Content Weight
        beta = .005   #Style Weight
        learning_rate = .01       
    
    content_weights['vgg_19/conv3/conv3_2/Conv2D:0'] = 1
    
    style_weights['vgg_19/conv1/conv1_2/Conv2D:0'] = 1
    style_weights['vgg_19/conv2/conv2_2/Conv2D:0'] = 1
    style_weights['vgg_19/conv3/conv3_3/Conv2D:0'] = 1
    style_weights['vgg_19/conv4/conv4_3/Conv2D:0'] = 1
    style_weights['vgg_19/conv5/conv5_3/Conv2D:0'] = 1
    
istr.main(model, 
          content_weights, 
          style_weights, 
          learning_rate,
          alpha, 
          beta, 
          cycles, 
          training_time, 
          start_jittering,
          stop_jittering, 
          jitter_freq, 
          display_image_freq, 
          max_image_dim = max_image_dim,
          random_initializer = random_initializer,
          use_wass = use_wass)
        
        
