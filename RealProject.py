
import cv2
import os
import random
import stat
import uuid
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


anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(30)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(15)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(30)

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
print(img.numpy().min())
plt.imshow(img)

'''
def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data

for file_name in os.listdir(os.path.join(ANC_PATH)):
    img_path = os.path.join(ANC_PATH, file_name)
    img = cv2.imread(img_path)
    augmentated_images = data_aug(img)

    for image in augmentated_images:
        cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())

for file_names in os.listdir(os.path.join(POS_PATH)):
    img_paths = os.path.join(POS_PATH, file_names)
    img = cv2.imread(img_paths)
    augmented_images = data_aug(img)

    for images in augmented_images:
        cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())
'''
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
data = data.shuffle(buffer_size=10000)

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

train_data = data.take(round(len(data)*.5))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

train_samples = train_data.as_numpy_iterator()
train_sample = train_samples.next()
print(train_samples.next())

print(train_data)

#this prints out 16 images because of the batch size. Gonna lower it though because i dont have a bunch on had


print(len(train_sample[0]))

# Gonna do the testing partition now
test_data = data.skip(round(len(data)*.5))
test_data = test_data.take(round(len(data)*.4))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)





#Embedding layer creation.
#L1 Distance Layer creation
#Compilation of the Siamese Network




#Building the embedding layer
def make_embedding():
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

embedding = make_embedding()

embedding.summary()

#Building the Distance Layer

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


l1 = L1Dist()



siamese_layer = L1Dist()


def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
siamese_model.summary()
