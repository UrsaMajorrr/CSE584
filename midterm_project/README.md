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

### Testing
In model_testing.py line 13, set the directory of the dataset that you want to read in. The datasets are stored in the datasets/ directory
In model_testing.py line 41, change the directory in the `torch.load()` function to be whatever model weights you want to test. Model weights are stored in the directory model_weights/

Finally, to run model_testing.py, in a terminal type `python model_testing.py`

### Dataset Curation
Dataset curation was tricky and it is not recommended to curate an entirely new dataset, but if it has to be done for continuation of this work the instructions are as follows.
First, you will need to set up an account with the APIs for some of the models. The APIs currently used are OpenAI, Mistral, Anthropic, AI21, and Cohere. The open source models are LLaMa 3.2 1B which can bet set up by following the instructions in the llama models GitHub. The GPT-NEO-1.3B can be accesssed through the Hugging Face transformers which can be installed with `pip install transformers`. Directions on how to use them are in Hugging Face.

After that you will have to understand how to query the LLMs and get their content. Examples can be found in llm_query.py in this repository. Make sure your dataset is saved as the name you want. It will be saved as JSON. In a python shell you can convert to csv using the pandas library (`df.to_csv("data_name"`). After this is set up you can call `torchrun --nproc-per-node=1 llm_query.py` from terminal if you have a LLaMa model or `python llm_query.py` from a terminal if you aren't using LLaMa models.
