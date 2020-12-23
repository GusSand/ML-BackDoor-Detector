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

class NoveltyDetector(object):
    
    def __init__(self, badNet_model_filepath, img_shape = [1, 55, 47, 3], \
                 clean_img_filepath = 'data/clean_validation_data.h5', num_class = 1283):
        """
        Parameters
        ----------
        badNet_model_filepath : 'models/XXXXXXXX_bd_net.h5'
        
        Return
        ----------
        None.
        
        """
        self.img_shape = np.array(img_shape, dtype=int)
        self.clean_img_filepath = clean_img_filepath
        self.num_class = num_class
        
        self.badNet_model_filepath = badNet_model_filepath
        
        
    def load_badnet_model(self):
        """
        Load BadNet model
        
        -------
        self.bdnet_model:
        
        """
        self.bdnet_model = load_model(self.badNet_model_filepath)
        
    
    def clean_img_classify_by_label(self):
        """
        Classify images by their .h5 data labels
        
        -------
        self.clean_img_list:
           list [label, images_set_in_this_label], 
           len=1283.
        
        """
        data = h5py.File(self.clean_img_filepath, 'r')
        img_data = np.array(data['data'])
        img_label = np.array(data['label'])
        img_data = img_data.transpose((0,2,3,1))
        
        clean_img_list = []
        for label_i in range(self.num_class):
            label_index = np.argwhere(img_label==label_i)

            clean_img_list.append(np.squeeze(img_data[label_index]))
            
        self.clean_img_list = clean_img_list
    
    
    def conv_4_result(self, img_set):
        """
        Get conv4 layer neure activation results
        
        Return
        -------
        conv_result.
        
        """
        conv4_index = 8
        # Extracts the outputs of the conv_4 layer:
        layer_outputs = [layer.output for layer in self.bdnet_model.layers[conv4_index-1:conv4_index]]
        # Creates a model that will return these outputs, given the model input:
        activation_model = models.Model(inputs=self.bdnet_model.input, outputs=layer_outputs)

        num_img = img_set.shape[0]
        
        layer_activation = activation_model.predict(img_set/255)
        layer_activation = layer_activation.reshape((num_img,layer_activation.shape[3]*layer_activation.shape[1]*layer_activation.shape[2]))
        
        return layer_activation
    
    
    def extract_clean_conv4_characters(self):
        """
        Extract clean data characters in conv4 layer
        
        -------
        self.conv4_characters_list:
            list [label, conv4_characters_for_one_label],
            len=1283.
        """
        
        self.clean_img_classify_by_label()
        self.load_badnet_model()
        
        self.conv4_characters_list = []
        print("***Start to extract clean data characters in conv4 layer***")
        for inx, img_array in enumerate(self.clean_img_list):
            result = self.conv_4_result(img_array)
            self.conv4_characters_list.append(result)
        print("***Finish***")

    
    def image_novelty_detector_predict(self, val_img_set):
        """
        Get the right predict label from image dataset
        
        Returns
        -------
        y_hat : if the image_i is poisoned, y_hat[i] = self.num_class+1
        
        """
        print("***Start bad net prediction***")
        y_hat = np.argmax(self.bdnet_model.predict(val_img_set/255), axis=1)
        i = 0;
        print("***Predict finish***")
        print("***Start novelty detector prediction***")
        result_conv4 = self.conv_4_result(val_img_set)
        
        for label_hat in y_hat:
            MATRIX = np.concatenate((self.conv4_characters_list[label_hat], result_conv4[i][None,:]), axis = 0)

            clf = LocalOutlierFactor(n_neighbors=(MATRIX.shape[0] - 2), algorithm='ball_tree')
            predict_result = clf.fit_predict(MATRIX)
            if(predict_result[-1] == -1):
                y_hat[i] = self.num_class
            
            i += 1
        print("***Predict finish***")
        print("*** Novelty Detector: Detected: {i}")
        return y_hat, result_conv4
    
    def get_conv4_characters_list(self):
        """
        Return conv4_characters_list
        
        Returns
        -------
        conv4_characters_list
        
        """
        return self.conv4_characters_list