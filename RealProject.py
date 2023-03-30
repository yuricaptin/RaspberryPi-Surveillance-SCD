
import cv2
import os
import random
import stat
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

#This file is for the structuring of the files in the path
#GPU Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#SSTEP 1
POS_PATH = os.path.join('Image', 'positive')
NEG_PATH = os.path.join('Image', 'negative')
ANC_PATH = os.path.join('Image', 'anchor')

#STEP 2 then deactivate
#os.makedirs(POS_PATH)
#os.makedirs(NEG_PATH)
#os.makedirs(ANC_PATH)

#Step 3.1
#To get around the win 5 error i used jupyter-lab and replaced the last string line for os.replace(Ex...) and replaced with NEW_PATH
pathways = 'C:/Users/lyork/source/repos/RealProject/lfw'

#Step 3.2 then deactivate
#for directory in os.listdir(pathways):
    #for file in os.listdir(os.path.join('C:/Users/lyork/source/repos/RealProject/lfw', directory)):
        #EX_PATH = os.path.join('C:/Users/lyork/source/repos/RealProject/lfw', directory, file)
        #NEW_PATH = os.path.join('C:/Users/lyork/source/repos/RealProject/RealProject/Image', file)
        #os.replace(EX_PATH, 'C:/Users/lyork/source/repos/RealProject/RealProject/Image')


anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(24)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(24)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(24)

ANC_PATH+'\*.jpg'

dir_test = anchor.as_numpy_iterator()

print(dir_test.next())


#Data Preprocessor is essential, will make this a seperate file later - Scaling and Resizing
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    return img

img = preprocess('Image\\anchor\\Austin3.jpg')
print(img.numpy().max())
plt.imshow(img)


#dataset.map(preprocess)

#Creation of the labelled dataset
#(anchor, postitive) => 1,1,1,1,1

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()

examp = samples.next()

#Gonna preprocess the stew out of these images

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

res = preprocess_twin(*examp)

plt.imshow(res[0])