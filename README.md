Basics of Deep Net: 
 1 . Local Receptive Region => 
 
 X , Y = > ( X - 1 , Y ) , ( X , Y - 1) , (X - 1 , Y - 1)
 
 
 ImageNet => Alex in 2011 proposes a model  which uses theses concepts to won the compitition. 
 VGG / Inception?
 
 2.  
 
 https://github.com/jacobgil/keras-cam
 
 dog ~900 images
 cat ~ 900 images
 top hat ~ 800 images
 woman ~ 1000 images
 .
 ...
 ..
 ..
 dinosaur ~ 900 images
 
 
 1 .  vgg_weights.h5 => Train a network archetecture ( VGG , Alexnet , Inception) for Image Classification for label.
 2 .  In Net Arch you have layers => input to softmax layer. Inbweteen layers => 
 
 Net Arch => [Conv, Pool, Conv, Pool, Conv-3,  Pool, Fully Connected-1,Fully Connected-2 (Class activation features), Softmax ]
 
  Fully Connected-1 => returns an array of dimension 4096 
  
  
  Image this an Image = > Conv -- 2046 * 2046 ( Channel - 1) 
  
  We have given VGG an image of a cat? Conv3 layer activations.
  

"""
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids

im = preprocess_image_batch(['examples/dog.jpg'], color_mode="bgr")

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=True)

" You will train your model"
model.compile(optimizer=sgd, loss='mse')

out = model.predict(im)

s = "n02084071"
ids = synset_to_dfs_ids(s)
heatmap = out[0,ids].sum(axis=0)

# Then, we can get the image
import matplotlib.pyplot as plt
plt.imsave("heatmap_dog.png",heatmap)

""
1 : git clone https://github.com/yardstick17/ConnectingDots
//2.  get vgg_weights.h5 and move them into training directory in ConnectingDots ( Make sure network archeitecture is same)
//3.  Train you fine tuned model.
2. move my data into the data directory. (cats, dogs, castles etc.)
3b. Train model with my dataset from scratch. Observe loss.

4. Make an operation on the image, so that an oil painting and normal image should be the same. normalizing an image. 
Cosign distance of the output images should be the same. 

GLCM : Grey Level Co-occurrence matrix

3c. f

(https://github.com/yardstick17/ConnectingDots/blob/master/neural_networks/convolutional_neural_network/training/image_classification.py)

2 : Load model with heatmap => True


Once you are done woth these steps = >image with heat map ready.

Next taask => Make bounding rectange around the area which is red (more activated).
  
git add submodule 
