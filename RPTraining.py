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
        progbar = tf.keras.utils.Progbar(len(data))

        # Creating a metric object
        r = Recall()
        p = Precision()

        for idx, batch in enumerate(data):

            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())

        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 50

train(train_data, EPOCHS)

test_input, test_val, y_true = test_data.as_numpy_iterator().next()

y_hat = siamese_model.predict([test_input, test_val])

[1 if prediction > 0.5 else 0 for prediction in y_hat]

print(y_true)

#Metrics calculation!
m = Recall()
m.update_state(y_true, y_hat)
m.result().numpy()


#Creation of the metric object
m = Precision()
m.update_state(y_true, y_hat)
m.result().numpy()

r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true, yhat)

print(r.result().numpy(), p.result().numpy())

plt.figure(figsize=(10,8))

plt.subplot(1,2,1)
plt.imshow(test_input[0])

plt.subplot(1,2,2)
plt.imshow(test_val[0])

plt.show()

#Saving the weights (Update the weights also works)
siamese_model.save('siamesemodelv2.h5')
L1Dist

#Reloading of the siamese model
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects = {'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

#Viewing the model summary
siamese_model.summary()
