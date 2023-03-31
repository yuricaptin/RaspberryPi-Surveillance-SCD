
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
    img = tf.image.resize(img, (105,105))
    img = img / 255.0
    return img

img = preprocess('Image\\anchor\\Austin3.jpg')
print(img.numpy().min())
plt.imshow(img)

#dataset.map(preprocess)

#Creation of the labelled dataset
#(anchor, postitive) => 1,1,1,1,1

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()

examp = samples.next()
print(examp)
#Gonna preprocess the stew out of these images

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

res = preprocess_twin(*examp)

plt.imshow(res[0])

print(res[2])

#Build the pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

#Prints out anchor image as well as the positive image
#VSCode never shows the plt so i will never see what the glorious photos are

sampss = data.as_numpy_iterator()
print(sampss.next())
print(len(sampss.next()))

yooo = sampss.next()
plt.imshow(yooo[1])


#We finally getting to the big Leagues!!
#Training partition!

print(round(len(data)*.7))

train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(8)
train_data = train_data.prefetch(4)

train_samples = train_data.as_numpy_iterator()
train_sample = train_samples.next()
print(train_samples.next())

print(train_data)

#this prints out 16 images because of the batch size. Gonna lower it though because i dont have a bunch on had


print(len(train_sample[0]))

# Gonna do the testing partition now
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(8)
test_data = test_data.prefetch(4)





#Embedding layer creation.
#L1 Distance Layer creation
#Compilation of the Siamese Network

inp = Input(shape=(100,100,3), name ='input_image')
print(inp)

#First Block
c1 = Conv2D(64, (10,10), activation='relu')(inp)
m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

    #Final FORM!
c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)


#Building the embussy layer
def make_embussy():
    inp = Input(shape=(100,100,3), name='input_image')

    #First Block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

    #Second Block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

    #Third Block
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

    #Final FORM!
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)


    return Model(inputs=[inp], outputs=[d1], name='embedding')

mod = Model(inputs=[inp], outputs=[d1], name='embedding')
print(mod)

embussy = make_embussy()

embussy.summary()

#Building the Distance Layer

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


l1 = L1Dist()

input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))

inp_embussy = embussy(input_image)
val_embussy = embussy(validation_image)

print(val_embussy)
print(embussy(input_image))
embussy(input_image)

siamese_layer = L1Dist()

print(siamese_layer(inp_embussy, val_embussy))

distances = siamese_layer(inp_embussy, val_embussy)

classifier = Dense(1, activation='sigmoid')(distances)

print(classifier)

def make_siamese_model():

    input_image = Input(name='input_img', shape=(100,100,3))
    
    validation_image = Input(name='validation_img', shape=(100,100,3))

    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embussy(input_image), embussy(validation_image))

    #Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
print(siamese_network)
siamese_network.summary()

siamese_model = make_siamese_model()
