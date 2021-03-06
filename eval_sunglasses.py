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
import cv2
from badnetcleaner import *


img_filename = str(sys.argv[1])
#model_filename = str(sys.argv[2])



def main():

    #img = matplotlib.image.imread(img_filename)
    img = cv2.imread(img_filename)
    
    img_list = []
    img_list.append(img)
    images = np.array(img_list)
    #images = images*255

    # Predict the poison data, label should be 1283 (N+1)
    bad_net_cleaner = BadNetCleaner('models/sunglasses_bd_net.h5','models/sunglasses_bd_weights.h5')
    y_hat = bad_net_cleaner.predict_label(images) # x_poison : image data X, MUST NOT /255!

    print(f"Result: Class ->{y_hat}")


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # config=tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess=tf.compat.v1.Session(config=config)
    # keras.__version__


    main()