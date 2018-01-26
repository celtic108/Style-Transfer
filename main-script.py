import inception_style_transfer as istr
import numpy as np

alpha = .16
beta = 10000
hours = 0.0
minutes = 6.0
training_time = 60*(60*hours + minutes)
display_image_freq=100
jitter_freq = 20
cycles=1000
stop_jittering= 1.
max_image_dim = 512                       

content_weights = {}
content_weights["InceptionV3/InceptionV3/Mixed_5c/concat:0"] = 1
    
style_weights = {}
style_weights["InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu:0"] = 1
        
style_weights["InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu:0"] = 1
style_weights["InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu:0"] = 1
style_weights["InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu:0"] = 1
style_weights["InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu:0"] = 1
        
style_weights["InceptionV3/InceptionV3/Mixed_5b/concat:0"] = 1
style_weights["InceptionV3/InceptionV3/Mixed_5c/concat:0"] = 1

pre_calc_style_grams, content_targets = istr.main(content_weights, 
                                                  style_weights, 
                                                  alpha, 
                                                  beta, 
                                                  cycles, 
                                                  training_time, 
                                                  stop_jittering, 
                                                  jitter_freq, 
                                                  display_image_freq, 
                                                  max_image_dim = max_image_dim)
        
        
