import tensorflow as tf
import keras
import os
import matplotlib as plt
import numpy as np
import random
import RealProject
import utils
from RealProject import *
from keras.metrics import Precision, Recall
from sklearn.metrics import confusion_matrix

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

EPOCHS = 10

@tf.function
def train_step(batch):

    #automatic recording of the inside of the neural network
    with tf.GradientTape() as tape:

        #Get anchot and positive/negative image
        X = batch[:2]
        #Getting Label
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



train(train_data, EPOCHS)

test_input, test_val, y_true = test_data.as_numpy_iterator().next()

y_hat = siamese_model.predict([test_input, test_val])

[1 if prediction > 0.8 else 0 for prediction in y_hat]

print(y_true)

#Metrics calculation!
m = Recall()
m.update_state(y_true, y_hat)
m.result().numpy()


#Creation of the metric object
n = Precision()
n.update_state(y_true, y_hat)
n.result().numpy()

r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true, yhat)



print(r.result().numpy(), p.result().numpy())

#Saving the weights (Update the weights also works)
siamese_model.save('siamesemodelv2.h5')
L1Dist

#Reloading of the siamese model
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects = {'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


siamese_model.predict([test_input, test_val])
#Viewing the model summary
t1 = siamese_model.summary()
print(t1)
plt.figure(figsize=(10,8))

plt.subplot(1,2,1)
plt.imshow(test_input[1])

plt.subplot(1,2,2)
plt.imshow(test_val[1])

plt.show()


#tp,fp,fn=utils.plot_confusion_matrix_from_data(y_hat,y_true,fz=18, figsize=(20,20), lw=0.5)

os.listdir(os.path.join('app_data', 'verify_image'))
os.path.join('app_data', 'input_image', 'input_image.jpg')

for image in os.listdir(os.path.join('app_data', 'verify_image')):
    validation_img = os.path.join('app_data', 'verify_image', image )
    print(validation_img)

def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('app_data', 'verify_image')):
        input_img = preprocess(os.path.join('app_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('app_data', 'verify_image', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('app_data', 'verify_image'))) 
    verified = verification > verification_threshold
    
    return results, verified

#OPEN CV INCOMING
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]
    
    cv2.imshow('Verification', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('v'):
    #This part will be modified with the xml file for the haar cascade to detect then do the siamese model.

        cv2.imwrite(os.path.join('app_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(siamese_model, 0.5, 0.5)
        print(verified)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()