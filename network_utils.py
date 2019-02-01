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

from os import remove, makedirs
from os.path import isdir
import tensorflow as tf

def restore_model(saver, model, sess):
    if model == 'Inception_V1':
        url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
        file_name = './input/inception_v1_2016_08_28.tar.gz'
        checkpoint = './input/inception_v1.ckpt'
    elif model == 'Inception_V3':
        url = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
        file_name = './input/inception_v3_2016_08_28.tar.gz'
        checkpoint = './input/inception_v3.ckpt'
    elif model == 'VGG_19':
        url = "http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz"
        file_name = './input/vgg_19_2016_08_28.tar.gz'
        checkpoint = './input/vgg_19.ckpt'
    else:
        raise Exception("Model name is not recognized. Please choose from ['Inception_V1', 'Inception_V3', 'VGG_19']")
    try:
        saver.restore(sess, checkpoint)
    except:
        ans = input("Model not found. Would you like to download it?: [y/n]")
        if ans in ['Y', 'y', 'Yes', 'YES', 'yes']:
            if not isdir('./input'):
                makedirs('./input')
            import urllib.request
            import shutil
            print("Downloading model")
            with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print("Extracting tar.gz")
            import tarfile
            tar = tarfile.open(file_name, 'r:gz')
            tar.extractall(path='./input/')
            tar.close()
            remove(file_name)
                    
            saver.restore(sess, checkpoint)

def gram(a):
    # Creates the TF operations to calculate the style gram of an image at a given level
    shape = a.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(a, shape=[-1,num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    length = tf.shape(matrix)[0]
    gram /= tf.cast(length, tf.float32)
    return gram



