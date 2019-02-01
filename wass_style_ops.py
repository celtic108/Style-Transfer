# Copyright (c) 2018 Vincent Marron | Released under MIT License

import tensorflow as tf


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
              