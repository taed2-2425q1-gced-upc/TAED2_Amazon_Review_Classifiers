"""
Extract and clean review text from labeled Amazon reviews.

This script processes a labeled dataset of Amazon reviews, where each review is prefixed with a 
label (e.g., '__label__<label>'). It extracts only the review text and saves the cleaned reviews, 
without labels, into a new file for further processing or analysis.

The script performs the following operations:
- Configures the root directory and adds it to the system path.
- Imports configurations and utility functions.
- Defines a function to parse and extract review text from labeled lines in an input file.
- Reads the file paths for the test and prediction datasets from a configuration file.
- Executes the `extract_reviews` function to save cleaned reviews to a specified output file.
"""

from pathlib import Path
import sys
from typing import TextIO

# Setting path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.config import RAW_DATA_DIR
from src import utilities

def extract_reviews(input_file: TextIO, output_file: TextIO) -> None:
    """
    Extracts review text from a labeled Amazon reviews file and writes the cleaned 
    reviews (without labels) to a new file.

    Args:
        input_file (TextIO): A text file containing labeled reviews in the format 
                             '__label__<label> <review_text>'.
        output_file (TextIO): The output file where reviews without labels are saved.
    
    """
    counter = 0
    # Iterate through each line in the input file
    for line in input_file:
        counter += 1
        # Split the line into label and review by the first space
        # and only take the review part (index 1)
        review = line.split(' ', 1)[1]
        # Write the review to the output file
        output_file.write(review)
        if counter == 100000:
            break

params = utilities.get_params(root_dir)

test_data_path: Path = RAW_DATA_DIR / params["test_dataset"]
predict_data_path: Path = RAW_DATA_DIR / params["predict_dataset"]

# Open the input and output files and run the extraction function
with open(test_data_path, 'r', encoding='utf-8') as infile,\
    open(predict_data_path, 'w', encoding='utf-8') as outfile:
    extract_reviews(infile, outfile)
