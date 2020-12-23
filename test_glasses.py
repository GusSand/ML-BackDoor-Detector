import sys
import h5py
import numpy as np
import tempfile
import os
import h5py
import matplotlib.pyplot as plt
# tensorflow, keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras import optimizers
from keras.models import load_model
from keras.preprocessing import image
from keras import models
# sklearn
from sklearn.neighbors import LocalOutlierFactor
import keras
import pdb
import matplotlib

from badnetcleaner import *
import unittest

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Predict the poison data, label should be 1283 (N+1)
        self.bad_net_cleaner = BadNetCleaner('models/sunglasses_bd_net.h5','models/sunglasses_bd_weights.h5')


    def test_clean(self):
        # Predict the clean test data, label should be y_label
        x_test, y_test = data_loader('data/clean_test_data.h5')

        y_hat_2 = self.bad_net_cleaner.predict_label(x_test) # x_test : image data X, MUST NOT /255!
        class_accu_2 = np.mean(np.equal(y_hat_2, y_test))*100
        print('Clean data accuracy:', class_accu_2)
        
    def test_backdoor(self):
        x_test, y_test = data_loader('data/sunglasses_poisoned_data.h5')

        y_hat = self.bad_net_cleaner.predict_label(x_test) # x_test : image data X, MUST NOT /255!
        class_accu = np.mean(np.equal(y_hat, 1283))*100
        print('Backdoor attack Success Rate for Sunglasses poisoned data:', 100 - class_accu)
        

if __name__ == '__main__':
        unittest.main()