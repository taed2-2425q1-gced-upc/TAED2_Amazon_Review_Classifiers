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
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         taed2_amazon_review_classifiers and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── taed2_amazon_review_classifiers   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes taed2_amazon_review_classifiers a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
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
3. Create virtual environment
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
5. Create custom kernel
```
pip install ipykernel
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
```
6. Start Jupyter
```
jupyter notebook
```
7. Select created kernel in the drop down menu top left when you have openend ```TAED2_Amazon_Review_Classifiers/notebooks/1.0-mhs-original-hugging-face-notebook.ipynb```
