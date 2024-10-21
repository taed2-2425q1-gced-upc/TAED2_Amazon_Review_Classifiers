"""
This module provides utility functions for loading parameters, tokenizers, and evaluation data
to be used in training or evaluating machine learning models, particularly for sentiment analysis.

Dependencies:
    - yaml
    - sys
    - pickle
    - pathlib.Path
    - tensorflow as tf

Functions:
    - get_params: Load parameters from a YAML configuration file.
    - get_tokenizer: Load a pre-trained tokenizer from a pickle file.
    - get_evaluation_file_lines: Read and return lines from a review evaluation file.
"""

import yaml
import sys
import pickle
from pathlib import Path
import tensorflow as tf

def get_params(root_dir: Path, param_file_name: str = "params.yaml") -> dict:
    """
    Load parameters from a YAML configuration file.

    Args:
        root_dir (Path): The root directory where the params file is located.
        param_file_name (str): The name of the YAML file containing parameters (default: "params.yaml").

    Returns:
        dict: A dictionary containing the parameters loaded from the YAML file.
    """
    # Setting path
    sys.path.append(str(root_dir))

    # Get parameter file
    params_path = root_dir / param_file_name

    # Load the YAML file
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    return params

def get_tokenizer(tokenizer_file_path: Path):
    """
    Load a pre-trained tokenizer from a pickle file.

    Args:
        tokenizer_file_path (Path): The path to the pickle file containing the tokenizer.

    Returns:
        The loaded tokenizer object from the pickle file.
    """
    # Load the tokenizer
    with open(tokenizer_file_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def get_evaluation_file_lines(evaluation_file_path: Path):
    """
    Read and return lines from an evaluation file containing Amazon reviews.

    Args:
        evaluation_file_path (Path): The path to the evaluation file containing review data.

    Returns:
        list[str]: A list of lines from the evaluation file, where each line is a string.
    """
    # Read reviews from evaluation file
    with open(evaluation_file_path, 'r', encoding='utf-8') as file:
        evaluate_file_lines = file.readlines()  # Read all lines from the file
    return evaluate_file_lines

def set_tokenizer(tokenizer_file_path: Path, tokenizer):
    """
    Save a pre-trained tokenizer to a pickle file.

    Args:
        tokenizer_file_path (Path): The path to where the pickle file containing
        the tokenizer should be saved.

    Returns:
        The path the tokenizer object was saved to.
    """
    # Save the tokenizer
    with open(tokenizer_file_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    return tokenizer_file_path