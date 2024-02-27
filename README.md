# CNN Car Type Detection

Dataset: https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped?select=Cars_Body_Type

A very simple car-type detection CNN built with Keras, in python. Main.py contains code to train the model, and test.py contains code to run a flask API server that takes in images, and returns predictions on those images.
Sadly, the trained .keras model file is way too large to commit to Github, so if you would like to run this, you'll have to train the model yourself. 

To do this, simply install all of the python packages, install the dataset from the link provided, replace the url in Main.py, and then run Main.py.
