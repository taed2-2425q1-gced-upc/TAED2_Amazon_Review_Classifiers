"""
Schemas module for validating input data using Pydantic.

This module defines the schema for the `PredictRequest` model used in the FastAPI application.
It ensures that the input review is a valid string and includes custom validation to check
the maximum length of the review.

Classes:
    - PredictRequest: A Pydantic model that validates the structure and content of the
    review provided for sentiment prediction.

Dependencies:
    - pydantic.BaseModel: The base class for data validation and settings
    management using Python type annotations.
    - typing.Any: A generic type hint used to represent any Python object.
"""

import sys
from pathlib import Path
from pydantic import BaseModel, field_validator

# Setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src import utilities

class PredictRequest(BaseModel):
    """
    A Pydantic model that represents the input schema for the review to be predicted.

    Attributes:
        review (str): The review text provided by the user.
        It must be a string of no more than 250 characters.
    """
    review: str

    @field_validator("review")
    @classmethod
    def check_string_length(cls, input: str) -> str:
        """
        Validator to ensure that the input review string does not exceed the
        maximum ammount of characters.

        Args:
            input (String): The value of the review field being validated.

        Raises:
            ValueError: If the length of the review exceeds the specified
            maximum amount of characters.

        Returns:
            String: The original value `input` if it passes the validation.
        """

        params = utilities.get_params(root_dir)

        if len(input) > params["max_review_length"]:
            raise ValueError(f"The input review exceeds the maximum length of \
                             {params['max_review_length']} characters.")
        return input
