import sys
import h5py
import numpy as np
import tempfile
import os
import h5py
import matplotlib.pyplot as plt
# tensorflow, keras
import tensorflow as tf
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

class New_Decision_Function(object):
    
    def __init__(self, badNet_weights_filepath, conv4_characters_list, img_shape = [1, 55, 47, 3], \
                 clean_img_filepath = 'data/clean_validation_data.h5', num_class = 1283):
        """
        Parameters
        ----------
        badNet_weights_filepath : 'models/XXXXXXX_bd_weights.h5'
        conv4_characters_list : result from extract_clean_conv4_characters
        
        Return
        ----------
        None.
        
        """
        self.img_shape = np.array(img_shape, dtype=int)
        self.clean_img_filepath = clean_img_filepath
        self.num_class = num_class
        self.conv4_characters_list = conv4_characters_list
        self.badNet_weights_filepath = badNet_weights_filepath
        
    def sub_model_net(self):
        """
        Sub_model_net structure.
        
        Return
        ----------
        small_model.
        
        """
        # define input
        x = keras.Input(shape=(960), name='input')
        fc_2 = keras.layers.Dense(160, name='fc_2')(x)
        add_1 = keras.layers.Activation('relu')(fc_2)
        drop = keras.layers.Dropout(0.5)
        # output
        y_hat = keras.layers.Dense(1283, activation='softmax', name='output')(add_1)
        model = keras.Model(inputs=x, outputs=y_hat)

        return model

    def load_weights_to_sub_model(self):
        """
        Sub_model_net load weights.
        
        Return
        ----------
        None.
        
        """
        
        sub_model = self.sub_model_net()
        sub_model.load_weights(self.badNet_weights_filepath, by_name=True)  
        
        return sub_model
    
    def retrain_sub_model(self):
        """
        Sub_model_net retrain.
        
        Return
        ----------
        None.
        
        """
        
        self.sub_model = self.load_weights_to_sub_model()
        X = np.array(self.conv4_characters_list)
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        y = np.repeat(np.arange(1283), 9)
        
        opt = optimizers.Adam(lr=0.001)
        self.sub_model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        print("***Start to creat new decision model***")
        self.sub_model.fit(X, y, epochs=20)
        print("***Finish***")
    
    def image_new_decision_function_predict(self, img_conv4_result, img_label):
        
        """
        New decision function prediction.
        
        Return
        ----------
        yhat : New decision function prediction results.
        
        """
        
        print("***Start new decision function prediction***")
        y_hat_submodel = np.argmax(self.sub_model.predict(img_conv4_result), axis=1)
        
        poison_index = np.where(y_hat_submodel != img_label)
        y_hat = y_hat_submodel
        y_hat[poison_index] = self.num_class
        print("***Predict finish***")
        
        return y_hat
        