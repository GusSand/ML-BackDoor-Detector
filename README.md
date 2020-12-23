# ML-BackDoor-Detector

We expect there to be two directories where you run the code:
1. models
2. data

the expected contents are below. Otherwise our scripts won't run. 

```bash
├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5

├── models
    └── anonymous_bd_net.h5
    └── anonymous_bd_weights.h5
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5

```

## I. Dependencies
   1. Python 3.6.9
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2



NOTES:

To run the code depending on the backdoor type:

- **python3 eval_sunglasses.py <test_image.png>**

- **python3 eval_multi.py <test_image.png>**

- **python3 eval_anonymous.py <test_image.png>**


Output should be either 1283 (if test_image.png is poisoned) or one class in range [0, 1282] (if test_image.png is not poisoned).



To run all the code do the following:
- Create the directories for the file either in google drive or locally
- open the notebook in colab or jupyter
- run from beginning to end. 

To run just parts here's the python code
```python 
  # First load the model and weights, this also gets the model cleaning the data
  bad_net_cleaner = BadNetCleaner('models/sunglasses_bd_net.h5','models/sunglasses_bd_weights.h5')

  # Load the data
  x_test, y_test = data_loader('data/clean_test_data.h5')

  # Get the label from both the Novelty Detector and the New Decision Function
  y_hat_2 = bad_net_cleaner.predict_label(x_test) # x_test : image data X, MUST NOT /255!
  class_accu_2 = np.mean(np.equal(y_hat_2, y_test))*100
  print('Classification accuracy:', class_accu_2)

```
