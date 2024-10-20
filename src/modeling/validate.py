"""
Module for validating an Amazon reviews dataset using Great Expectations.

This script reads review data (label and review text) from a file, processes it,
and runs a series of validation checks using Great Expectations. The validations
ensure the data adheres to specific expectations (e.g., columns exist, labels are correct),
and the results are logged and visualized using DagsHub and Great Expectations.
"""

import sys
import typing
from pathlib import Path
import numpy as np
import pandas as pd
import typer
from loguru import logger
import dagshub
import great_expectations as gx

# Setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import RAW_DATA_DIR

# Initialize Typer app for command-line interface
app = typer.Typer()

# Initialize a Great Expectation Context
context = gx.get_context()

# Initialize DagsHub integration
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

def split_reviews_labels(input_lines: list[str]) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Split raw input lines into reviews and their corresponding labels.

    Args:
        input_lines (list[str]): List of lines containing review data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays,
        one for reviews and one for labels (1 for positive, 0 for negative).
    """
    labels = []
    reviews = []

    for line in input_lines:
        split_line = line.strip().split(' ', 1)
        label = 1 if split_line[0] == '__label__2' else 0
        review = split_line[1]
        labels.append(label)
        reviews.append(review)

    # Convert reviews and labels to numpy arrays
    reviews = np.array(reviews)
    labels = np.array(labels)

    return reviews, labels

@app.command()
def main(
    input_data_path: Path = RAW_DATA_DIR / "development.txt",
):
    """
    Main function to validate the Amazon reviews dataset.

    This function reads raw review data from the specified file, processes the data
    to extract labels and review texts, and runs validations using Great Expectations.
    Validation results are logged and visualized via DagsHub and Great Expectations.

    Args:
        input_data_path: Path to the input text file containing review data without headers.
    """

    logger.info(f"Using Great Expectations to validate \
                data from {input_data_path}")

    # Read reviews from evaluation file
    with open(input_data_path, 'r', encoding='utf-8') as file:
        evaluate_file_lines = file.readlines()  # Read all lines from the file


    # Split reviews and labels from the input data
    reviews, labels = split_reviews_labels(evaluate_file_lines)

    # Create Dataset with explicit column names
    input_data = pd.DataFrame({
        'Reviews': reviews,
        'Labels': labels
    })

    # Create Suite
    suite_name = "amazon_review_classifier_suite"
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)

    # Create expectations
    column_names_list_exp = gx.expectations.ExpectTableColumnsToMatchOrderedList(
        column_list=["Reviews", "Labels"]
    )
    distinct_labels_exp = gx.expectations.ExpectColumnDistinctValuesToContainSet(
        column="Labels",
        value_set=[0, 1]
    )
    not_null_reviews_exp = gx.expectations.ExpectColumnValuesToNotBeNull(column="Reviews")
    not_null_lables_exp = gx.expectations.ExpectColumnValuesToNotBeNull(column="Labels")

    # Add Expectations
    suite.add_expectation(column_names_list_exp)
    suite.add_expectation(distinct_labels_exp)
    suite.add_expectation(not_null_reviews_exp)
    suite.add_expectation(not_null_lables_exp)

    # Data source
    data_source_name = "to_validate"
    data_source = context.data_sources.add_pandas(name=data_source_name)

    # Data Asset
    data_asset_name = "to_validate_data_asset"
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)

    # Batch Definition
    batch_definition_name = "to_validate_batch_definition"
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        batch_definition_name
    )

    # Provide dataframe to batch
    batch_parameters = {"dataframe": input_data}

    batch_definition = (
        context.data_sources.get(data_source_name)
        .get_asset(data_asset_name)
        .get_batch_definition(batch_definition_name)
    )

    # Create a Validation Definition
    validation_def_name = "my_validation_definition"
    validation_definition = gx.ValidationDefinition(
        data=batch_definition, suite=suite, name=validation_def_name
    )

    # Add the Validation Definition to the Data Context
    validation_definition = context.validation_definitions.add(validation_definition)

    # Test the Expectation
    validation_results = validation_definition.run(batch_parameters=batch_parameters)
    logger.info(validation_results)
    print(validation_results)

    logger.info(f"Validated {len(reviews)} reviews with Great Expectations.")

if __name__ == "__main__":
    app()
