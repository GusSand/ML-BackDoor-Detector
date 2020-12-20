# ML-BackDoor-Detector


To run the code you need to use it as follows:

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
