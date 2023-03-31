import tensorflow as tf
import keras
import os
import matplotlib as plt
import numpy as np
import random
import RealProject
from RealProject import *
from RPTraining import *
from keras.metrics import Precision, Recall

test_input, test_val, y_true = test_data.as_numpy_iterator.next()

y_hat = siamese_model.predict([test_input, test_val])

[1 if prediction > 0.5 else 0 for predicion in y_hat]

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


