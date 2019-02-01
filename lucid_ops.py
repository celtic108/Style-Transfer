# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
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

#RegisterGradients added by celtic108

import tensorflow as tf
import numpy as np

def get_fourier_features(content_image):
            ccss = np.array([[0.26, 0.09, 0.02],
                            [0.27, 0.00, -0.05],
                            [0.27, -0.09, 0.03]]).astype("float32")
            mnss = np.max(np.linalg.norm(ccss, axis=0))
            print("MNSS: ", mnss)
            corr_matrix = (ccss / mnss).T
            def rfft2d_freqs(h,w):
                fy = np.fft.fftfreq(h)[:,None]
                if w % 2 == 1:
                    fx = np.fft.fftfreq(w)[: w // 2 + 2]
                else:
                    fx = np.fft.fftfreq(w)[: w // 2 + 1]
                return np.sqrt(fx * fx + fy * fy)
        
            sd = 0.01
            batch, h, w, ch = content_image.shape
            freqs = rfft2d_freqs(h, w)
            init_val_size = (2, ch) + freqs.shape
            init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
            scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h))
            scale *= np.sqrt(w*h)
            return init_val, scale, corr_matrix


@tf.RegisterGradient("Custom1")
def redirected_relu_grad(op, grad):
  assert op.type == "Relu"
  x = op.inputs[0]

  # Compute ReLu gradient
  relu_grad = tf.where(x < 0., tf.zeros_like(grad), grad)

  # Compute redirected gradient: where do we need to zero out incoming gradient
  # to prevent input going lower if its already negative
  neg_pushing_lower = tf.logical_and(x < 0., grad > 0.)
  redirected_grad = tf.where(neg_pushing_lower, tf.zeros_like(grad), grad)

  # Ensure we have at least a rank 2 tensor, as we expect a batch dimension
  assert_op = tf.Assert(tf.greater(tf.rank(relu_grad), 1), [tf.rank(relu_grad)])
  with tf.control_dependencies([assert_op]):
    # only use redirected gradient where nothing got through original gradient
    batch = tf.shape(relu_grad)[0]
    reshaped_relu_grad = tf.reshape(relu_grad, [batch, -1])
    relu_grad_mag = tf.norm(reshaped_relu_grad, axis=1)
  result_grad = tf.where(relu_grad_mag > 0., relu_grad, redirected_grad)

  global_step_t =tf.train.get_or_create_global_step()
  return_relu_grad = tf.greater(global_step_t, tf.constant(16, tf.int64))

  return tf.where(return_relu_grad, relu_grad, result_grad)

@tf.RegisterGradient("Custom2")
def redirected_relu6_grad(op, grad):
  assert op.type == "Relu6"
  x = op.inputs[0]

  # Compute ReLu gradient
  relu6_cond = tf.logical_or(x < 0., x > 6.)
  relu_grad = tf.where(relu6_cond, tf.zeros_like(grad), grad)

  # Compute redirected gradient: where do we need to zero out incoming gradient
  # to prevent input going lower if its already negative, or going higher if
  # already bigger than 6?
  neg_pushing_lower = tf.logical_and(x < 0., grad > 0.)
  pos_pushing_higher = tf.logical_and(x > 6., grad < 0.)
  dir_filter = tf.logical_or(neg_pushing_lower, pos_pushing_higher)
  redirected_grad = tf.where(dir_filter, tf.zeros_like(grad), grad)

  # Ensure we have at least a rank 2 tensor, as we expect a batch dimension
  assert_op = tf.Assert(tf.greater(tf.rank(relu_grad), 1), [tf.rank(relu_grad)])
  with tf.control_dependencies([assert_op]):
    # only use redirected gradient where nothing got through original gradient
    batch = tf.shape(relu_grad)[0]
    reshaped_relu_grad = tf.reshape(relu_grad, [batch, -1])
    relu_grad_mag = tf.norm(reshaped_relu_grad, axis=1)
  result_grad =  tf.where(relu_grad_mag > 0., relu_grad, redirected_grad)

  global_step_t = tf.train.get_or_create_global_step()
  return_relu_grad = tf.greater(global_step_t, tf.constant(16, tf.int64))

  return tf.where(return_relu_grad, relu_grad, result_grad)
