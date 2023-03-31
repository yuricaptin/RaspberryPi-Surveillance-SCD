import tensorflow as tf
import keras
import os
import matplotlib as plt
import numpy as np
import random
import RealProject
from RealProject import *
from keras.metrics import Precision, Recall

#This py is for training the model
#Loss and Optimizer

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

#I will now lay waste to the land of checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model = siamese_model)

#Train Step Function
test_batch = train_data.as_numpy_iterator()
batch_1 = test_batch.next()
print(len(batch_1))



@tf.function
def train_step(batch):

    #automatic recording of the inside of the neural network
    with tf.GradientTape() as tape:

        X = batch[:2]
        
        y = batch[2]

        yhat = siamese_model(X, training=True)

        loss = binary_cross_loss(y, yhat)
    print(loss)

    grad = tape.gradient(loss, siamese_model.trainable_variables)

    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss


def train(data, EPOCHS):

    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(train_data))

        # Creating a metric object
        r = Recall()
        p = Precision()

        for idx, batch in enumerate(train_data):

            train_step(batch)
            progbar.update(idx+1)

        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 50

train(train_data, EPOCHS)