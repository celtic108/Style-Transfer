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

from PIL import Image
from os.path import isdir
from os.path import join as osjoin
from os import makedirs, walk
import numpy as np
from random import randint

def display_image(gen_pixels, save=True, name=0):
    # Uses PIL Image to display images periodically during optimization
    if gen_pixels.shape[0]==1:
        height = gen_pixels.shape[1]
        width = gen_pixels.shape[2]
    else:
        height = gen_pixels.shape[0]
        width = gen_pixels.shape[1]
    pix = Image.fromarray(np.clip(gen_pixels[0],0,255).astype('uint8'))
    pix.show()
    if save:
        if not isdir('./trial_runs/'):
            makedirs('./trial_runs/')
        pix.save("./trial_runs/"+str(name)+".jpg")


def load_images(input_dir, max_image_dim = float('inf'), target_shape = None):
    for root, dirs, files in walk(input_dir):
        for file in files:
            try:
                image = Image.open(osjoin(root, file)).convert("RGB")
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
                    w_diff = max(0, w-target_shape[1])
                    h_diff = max(0, h-target_shape[2])
                    image = image.resize((max(target_shape[1], w), max(target_shape[2], h)), resample = Image.LANCZOS)
                    offset_x = randint(0,w_diff)
                    offset_y = randint(0,h_diff)
                    image = image.crop((offset_x, offset_y, target_shape[1] + offset_x, target_shape[2] + offset_y))
                return np.expand_dims(np.array(image), axis=0)
            except:
                print("Found file that was not an image. Keep looking.")

def preprocess(image):
    image = ((image/255.)-0.5)*2
    #image = image/255.
    return image
    
def un_preprocess(image):
    #return 255.*image
    return 255.*(image +1)/2.

