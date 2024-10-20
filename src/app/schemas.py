"""
Schemas module for validating input data using Pydantic.

This module defines the schema for the `PredictRequest` model used in the FastAPI application.
It ensures that the input review is a valid string and includes custom validation to check
the maximum length of the review.

Classes:
    - PredictRequest: A Pydantic model that validates the structure and content of the review provided
                      for sentiment prediction.

Dependencies:
    - pydantic.BaseModel: The base class for data validation and settings management using Python type annotations.
    - typing.Any: A generic type hint used to represent any Python object.
"""

from pydantic import BaseModel, field_validator

class PredictRequest(BaseModel):
    """
    A Pydantic model that represents the input schema for the review to be predicted.

    Attributes:
        review (str): The review text provided by the user. It must be a string of no more than 250 characters.
    """
    review: str

    @field_validator("review")
    @classmethod
    def check_string_length(cls, input: str) -> str:
        """
        Validator to ensure that the input review string does not exceed 250 characters.

        Args:
            input (String): The value of the review field being validated.

        Raises:
            ValueError: If the length of the review exceeds 250 characters.

        Returns:
            String: The original value `input` if it passes the validation.
        """
        if len(input) > 250:
            raise ValueError("The input review exceeds the maximum length of 250 characters.")
        return input
