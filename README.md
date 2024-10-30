# TAED2_Amazon_Review_Classifiers

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Sentiment Classification for Amazon reviews

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Holds our measured emission files as well as the GAISSA label files
│   ├── emissions      <- Holds measured emission files.
│   └── gaissa_labels  <- Holds the created Gaissa Labels.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── params.yaml	       <- Contains parameter values for tweaking the model as well as file paths
│                       
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│   └── coverage       <- Generated pytest coverage report
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes taed2_amazon_review_classifiers a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── utilities.py            <- Holds common utility functions
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py                       <- Code to run model inference with trained models          
    │   ├── evaluate.py                      <- Code to evaluate the performance of model against unseen data
    │   ├── check_data.py                    <- Code to check the data
    │   ├── train_test_split.py              <- Code to split a dataset into train and test data
    │   ├── train.py                         <- Code to train models
    │	├── tokenization_and_split.py        <- Code to split and tokenize the data
    │	├── embedding_matrix_generation.py   <- Code to generate the embedding matrix
    │	└── model_test.py                    <- Code to test the model
    │
    └── app
        ├── api.py              <- Code for FastAPI interface
        └── schema.py           <- Code for input validation using pydantic
└── tests   <- Pyest test files to test model and code
```

## Energy Label

### Training

![GAISSA Training Label](docs/gaissa_labels/Gaissa_training_label_sentiment.png)

![Training Label PDF](docs/gaissa_labels/Gaissa_training_label_sentiment.pdf?raw=true "Training Label")

### Inference (100k Reviews)
![GAISSA Inference Label](docs/gaissa_labels/Gaissa_inference_label_sentiment.jpg)

![Inference Label PDF](docs/gaissa_labels/Gaissa_inference_label_sentiment.pdf?raw=true "Inference Label")


## Setup Guide (Ubuntu)
1. Clone repository
```
git clone <repository-url>
```
2. Get correct Python version & Jupyter
```
sudo apt-get install python3.10
sudo apt-get install jupyter
```
3. Create virtual environment (optional)
```
pip install virtualenv
cd  <project root>
python3 -m venv venv
source venv/bin/activate
```
4. Install packages
```
pip install -r requirements.txt
```
## Amazon Reviews Sentiment Analysis Pipeline

This repository contains a pipeline for a machine learning-based sentiment analysis project on Amazon reviews. The pipeline processes labeled review text data, trains a model, evaluates its performance, and allows for predictions on new data. Each step is modular, enabling automation and retraining with new data as needed. Pipeline parameters are configurable through the `params.yaml` file.

### Pipeline Steps

To run each step in the pipeline, follow the instructions below. All modules can be executed using the command `python3 -m src.modeling.<filename>`, as shown in each step.

#### 1. Data Splitting (`train_test_split.py`)

This script splits the raw dataset into separate training and test sets, allowing for generalization of the model to unseen data.

```bash
# Run the data splitting script
python3 -m src.modeling.train_test_split
```

#### 2. Tokenization and Data Preparation (`tokenization_and_split.py`)
This script tokenizes the training data and splits it into sequences for model training and validation.

```bash
# Run the tokenization and data preparation script
python3 -m src.modeling.tokenization_and_split
```

#### 3. Embedding Matrix Generation (`embedding_matrix_generation.py`)
This script creates an embedding matrix from a pre-trained word embedding model, enabling the model to leverage semantic relationships between words.

```bash
# Run the embedding matrix generation script
python3 -m src.modeling.embedding_matrix_generation
```

#### 4. Model Training (`train.py`)
This script trains the sentiment analysis model using the tokenized data, labels, and the embedding matrix.

```bash
# Run the model training script
python3 -m src.modeling.train
```

#### 5. Model Evaluation (`evaluate.py`)
This script evaluates the final trained model on the test dataset to assess its accuracy and generalization.

```bash
# Run the model evaluation script
python3 -m src.modeling.evaluate
```

For prediction, use the `predict.py` or `predict_emissions.py` modules in the same way after the model is trained.

#### Script for automatic execution

```
#!/bin/bash

# Run each step in sequence
python3 -m src.modeling.train_test_split
python3 -m src.modeling.tokenization_and_split
python3 -m src.modeling.embedding_matrix_generation
python3 -m src.modeling.train
python3 -m src.modeling.evaluate
```

## API Guide
1. After performing steps 1.-4. from the Setup Guide, set up the server. Specify the port as needed. 
   ```
   uvicorn src.app.api:app     --host 0.0.0.0     --port 5000     --reload     --reload-dir src/app     --reload-dir models
   ```
2. When the server is set up, you can interact with the API either by putting the ip adress of the machine or localhost + the port in your browser, depending where you set up the API.
3. Currently two endpoints are supported: predictReview & predictReviews. As the name implies, is one for single reviews and the other for a batch of reviews.
4. The JSON structures look like the following:
   ```
   # For a single review
      {
        "review": "This is single good review."
      }
    ```
    ```
    # For multiple reviews
    [
      {
        "review": "This is a good review."
      },
      {
        "review": "This is also a good review."
      },
      {
        "review": "This is a bad review."
      }
    ]
    ```
5. There is a validation in place to check that no review exceeds a current maximum of 250 words (which is configurable in the params file)
