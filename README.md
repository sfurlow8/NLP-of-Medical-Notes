# Overview

The files here are adapted from [Marta Fernandes et al. 2020](https://medinform.jmir.org/2021/2/e25457/). The files run a Natural Language Processing pipeline on doctor-written EHR notes of ICU patients. We take the raw text of notes, process them into computer-readable embeddings, and use those tokens as input into a logistic regression model that predicts a new patient’s coma recovery outcome (CPC score).

The pipeline is executed in a Docker container (setup described below). The files assume a file structure with 3 main directories: `code`, `data`, and `results`. The code files should be executed from within the `code` directory. Data files are stored in `data`, and processed data files and plots are created in `results`.

## Preprocessing functions

The preprocessing functions are `ds_prep`, `join_fields`, and `lemma` in `preprocessing_functions.py`. 

`ds_prep` processes an unstructured chunk of note text into a reduced form of the original note that contains only pertinent information. Pertinent information is defined as the 200 characters following each of several section headings shown in the function: principal diagnosis, code status, surgeries, tests, allergies, diet, medical history, findings from studies, follow-up care, activity, physical exam, neurological exam, physical/occupational therapy, progress review, treatments, labs, and hospital course. Any information about discharge instructions is left out to avoid the model learning recovery scores from explicit discharge descriptions. 

`join_fields` takes all the sections extracted from the note by `ds_prep` and unifies them into a single reduced version of the original note. When join_fields is called, the field argument should be the column name where the raw note text is stored in your data. The flag argument specifies what sections you want to include in your reduced note. Unless you are working with discharge summaries, the flag should be ‘noDD’ to not include any sections that may have explicit discharge descriptions. 

`lemma` uses Python’s `nltk WordNet lemmatizer` to convert all the words in the reduced note to their lemmas (the root form of each word). This ensures that words with the same meaning in different tenses are treated as the same feature during modeling. `word_tokenize` from the `nltk` package is used to tokenize the reduced note into unigrams, bigrams, and trigrams. `word_tokenize` uses a Bag of Words model for tokenization, paying no attention to the order and position of words in a note for simplicity and speed. At the end of lemma, some common typos are corrected and common abbreviations are expanded into their unabbreviated form. 

## Modeling functions

The functions in `modeling_functions.py` feed preprocessed notes into a logistic regression model that predicts coma recovery outcome. Plots describing performance and feature importance are generated in the `save_path` specified at the beginning of the file, which should be `results` or a subdirectory of `results`.  

`train_test_encode` divides the given data into train and test (0.3) sets stratified by CPC score and encodes the CPC score labels.

`ngram` uses `sklearn`'s `CountVectorizer` to create vectors of TF-IDF vectors for tokens in our training and test sets of reduced notes. 

`modeling` creates a `LogisticRegression` classifier with a `OneVsRest` scheme and LASSO shrinkage using `sklearn`. We use 5 fold cross validation to select hyperparameters. Only tokens present in 10% or more of all the notes are included for modeling. 

The remaining plotting functions show model performance and influential tokens on model predictions. 


## Main

All the files in the pipeline assume that the note text to be processed and modeled is stored under the column name `deid_note` and the ground truth CPC scores of patients are stored under the column name `cpc_at_discharge`. The user must specify `path` where their data is coming from and `path_` where their resulting plots will be generated. The user must also specify the title of their data file and how many CPC groupings they would like to predict. 

# Using the pipeline

## Docker setup

Run the Docker Daemon by opening Docker Desktop on your computer. In a terminal window, navigate to the directory that contains the Dockerfile. To build an image from the Dockerfile, enter the following command with 'name' replaced with the name you want to give your image:

    docker build -t name:latest .

Use the command `docker images` to view your images and make sure that your new image has been created with the intended name. Copy your image's IMAGE ID for later. To run your image and enter a Docker container:

    docker run -it name:latest /bin/bash

In the container, make a workspace folder that contains the same directories on your local machine for consistency:

    mkdir workspace
    cd workspace
    mkdir code data results

Exit the Docker container with `exit`. 

## Training and testing a model

The bash scripts `copy_files.sh` and `copy_plots.sh` transfer files back and forth between your local machine and the Docker container where the pipeline will be run. Replace the IMAGE ID in these 2 scripts with the IMAGE ID from the image you created, and replace the names of the data files with the names of your data files. On your local machine, navigate to the `code` folder. To copy your working files to the Docker container:

        bash copy_files.sh

Then enter the Docker container wih the docker run command from above, navigate to workspace/code, and run:

        python main.py

Outside of the Docker container, in the `code` folder on your local machine, copy the resulting plots generated in the Docker container to your local machine so you can view them:

        bash copy_plots.sh

If you have successfully run the pipeline and copied your files, the performance plots will appear in the results folder on your local machine. 
