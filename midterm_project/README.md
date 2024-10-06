This readme highlights the necessary dependencies and how to run the code to reproduce results

## Dependencies
PyTorch with Cuda (latest version)
sklearn==1.3.2
transformers==4.45.1
pandas==2.1.4

## Result Reproductions
### Training
In model_train.py line 20, change the directory to the dataset that you would like to train the model on. The dataset directory is datasets/
In model_train.py line 100, change the name of the model that you training to whatever you desire. The weights of the model will be saved under this name and will be needed to load them in when testing

Finally, to run model_train.py, in a terminal type `python model_train.py`
