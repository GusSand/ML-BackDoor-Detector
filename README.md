# ML-BackDoor-Detector


To run the code you need to use it as follows:

```python 
  # First load the model and weights, this also gets the model cleaning the data
  bad_net_cleaner = BadNetCleaner('models/sunglasses_bd_net.h5','models/sunglasses_bd_weights.h5')

  # Load the data
  x_poison, y_poison = data_loader('data/sunglasses_poisoned_data.h5')

  # Get the label from both the Novelty Detector and the New Decision Function
  y_hat = bad_net_cleaner.predict_label(x_poison) # x_poison : image data X, MUST NOT /255!
  class_accu = np.mean(np.equal(y_hat, 1283))*100
  print('Classification accuracy:', class_accu)
```
