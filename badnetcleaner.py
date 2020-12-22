# input is 'models/XXXXXXXX_bd_net.h5', 'models/XXXXXXXX_bd_weights.h5', 'data/clean_validation_data.h5'
from New_Decision_Function import *
from NoveltyDetector import *

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data, y_data
    
class BadNetCleaner(object):
    
    def __init__(self, badNet_model_filepath, badNet_weights_filepath):
        """
        Parameters
        ----------
        badNet_model_filepath : 'models/XXXXXXXX_bd_net.h5'
        badNet_weights_filepath : 'models/XXXXXXX_bd_weights.h5'
        
        Return
        ----------
        None.
        
        """
        self.badNet_model_filepath = badNet_model_filepath
        self.badNet_weights_filepath = badNet_weights_filepath
        #pdb.set_trace()         
        self.novelty_detector = NoveltyDetector(self.badNet_model_filepath)
        self.novelty_detector.extract_clean_conv4_characters()
        self.conv4_characters_list = self.novelty_detector.get_conv4_characters_list()
        self.new_decision_function = New_Decision_Function(self.badNet_weights_filepath, self.conv4_characters_list)
        self.new_decision_function.retrain_sub_model()
        print("***Initialzation finish***")

    import pdb

    def predict_label(self, img_set):
        """
        Parameters
        ----------
        img_set : image data X, MUST NOT /255!
        
        Return
        ----------
        y_hat_2 : BadNetCleaner predict results.
        
        """
        
   
        print("**************************************************************")
        y_hat, img_conv4_result = self.novelty_detector.image_novelty_detector_predict(img_set)
        y_hat_2 = self.new_decision_function.image_new_decision_function_predict(img_conv4_result, y_hat)
        print("**************************END*********************************")
        
        return y_hat_2
