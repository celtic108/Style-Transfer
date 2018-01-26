# Style-Transfer
Style Transfer using Tensorflow and Inception V3 Network
Influenced by https://github.com/hwalsuklee/tensorflow-style-transfer

Requires Tensorflow and pretrained weights for the Inception V3 network. 

Create a folder "contentpicture" to store the desired content image.

Create a folder "stylepicture" to store the desired style image.

Create a folder "trialruns" to capture the finished images.

Further work:
- Improve the way the network is loaded to remove unneeded layers
- Add a de-noising term to reduce high frequency artifacts - See https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/15_Style_Transfer.ipynb
- Experiment with other network architectures like VGG
