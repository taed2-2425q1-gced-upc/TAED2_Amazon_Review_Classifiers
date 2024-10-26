""" Extracts review text from a labeled Amazon reviews file and writes the cleaned reviews (without labels) to a new file. """
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
with open(test_data_path, 'r') as infile, open(predict_data_path, 'w') as outfile:
    extract_reviews(infile, outfile)