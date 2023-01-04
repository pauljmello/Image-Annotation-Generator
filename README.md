# CMPE258 Project: Image Captioning

## Environment Setup

There are two options for environmental setup, a Conda environment or the requirements.txt. A conda environment is recommended as a specific python version was used.  

### Conda Environment

1. Go to root directory of project in terminal
2. Run `conda env create -f caption_env.yml`
3. Confirm caption_env environment created with `conda env list`

### requirements.txt

1. Go to root directory of project in terminal
2. Run `pip install requirements.txt`

## Generating Captions

Caption generation is done from the command line. The image to be captioned should be accessable from the root directory of the project. 

1. Go to root directory of project and activate Conda environment if necessary
2. Run `python generate_caption.py --p PHOTO_PATH --m MODEL_NAME`
  - PHOTO_PATH is the path from the directory to the desired photo. Ex: manedwolf.jpg
  - MODEL_NAME is the name of one of the trained models
    - base = base  model
    - selu = use of selu activation function rather than relu
    - dropout = use of 0.2 droout rather than 0.5
    - layers = extra layer in the feature extraction and decoder
    - adamax = use of adamax optimizer instead of adam

## Full Training Process
Files were run either by pressing the run button in Pycharm or with `python FILE` in the terminal. The files were run in the following order to train the model:
1. prep_dataset
2. prep_text
3. train_model
4. evaluate_model
5. save_tokenizer

Any ablations done to train_model is then followed by running train_model again.

## Photo Dataset Cleaning
For out project we used the Flickr8K dataset to train the model. However, there were issues with the frequency of terms. For example, an abundance of the word 'red' in reference to shirts would have all captions with shirts have the word 'red'. Thus, we cleaned up the dataset by first counting the number of occurences of adjectives in the dataset. Any word in a certain category, like color, that occured inproportionately frequently was then altered. Many caption with those colors in it had the word removed so that the frequency of colors were more even without making the captions inaccurate.

## Using Different Models to Caption
If you posses another model you wish to generate captions with, simply change line 100 in generate_caption.py. The line should be as follows:  
`model = load_model(MODEL_PATH)` where MODEL_PATH is replaced with the path to the h5 file such as 'models/new-model.h5'.  
It can then be run in the command line with `python generate_model --p PHOTO_PATH`  
  
## Verification Against Baseline
Any new models trained can be verified against the base model. This model is found in 'models/model-tf2.h5'. This model is also run with the `--m base` flag if the code is not altered as per the previous section.
