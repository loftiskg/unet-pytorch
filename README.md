## model.py
*  Contains the unet model. **The implementation of the unet architecture is not my own.  This unet implementation was built by jaxony.  See his repository for more details. https://github.com/jaxony/unet-pytorch**

## data.py
* contains the dataset class that I use to test the architecture.

## test_data_genereator.py
* script that creates and saves images/label pairs of a specified size with random patches of the image blurred.  The label of a respective image corresponds with blur area of that image.  These images are the test images used by data data.py.  This is admitly a little buggy, but it does the trick for a simple test to see if the model converges.

## train.py
* trains the model using the test images and performs inference.  Saves the predicted masks to file.  Has the option to save the model weights.

