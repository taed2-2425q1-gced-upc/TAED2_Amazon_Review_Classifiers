"""
This module contains tests for the robustness of the sentiment analysis model against random 
word substitutions in product reviews. It utilizes pytest to evaluate whether the model's 
predictions are consistent when minor alterations are made to the text.

Tests include:
- Assessing the model's response to pairs of original and altered reviews, which maintain 
  the overall structure but substitute key words with unrelated terms.
  - Reviews that express a positive sentiment.
  - Reviews that express a negative sentiment.

The tests check if the sentiment prediction remains consistent (conserved) after the word 
substitutions, which is crucial for determining the model's ability to focus on relevant 
features of the text.

Fixtures:
- `model_and_tokenizer`: A pytest fixture that loads the pre-trained sentiment analysis model 
  and tokenizer from the specified directories, ensuring that the tests run in a controlled 
  environment.

To run the tests, execute the module directly or use pytest from the command line.

Usage:
    python -m pytest path_to_this_file.py
"""

from pathlib import Path
import sys
import tensorflow as tf
import pytest
import dagshub

# Setting path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.config import MODELS_DIR, RESOURCES_DIR
from src import utilities
from src.modeling.predict import predict_sentiment

# Initialize DagsHub integration with MLflow
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

def check_prediction_conservation(sentiment_1, sentiment_2):
    pred_1 = "positive" if sentiment_1 >= 0.5 else "negative"
    pred_2 = "positive" if sentiment_2 >= 0.5 else "negative"
    return pred_1 == pred_2

# Load the model and tokenizer
@pytest.fixture(scope="module")
def model_and_tokenizer():
    utilities.check_tensorflow_version()
    params = utilities.get_params(root_dir)

    model_path: Path = MODELS_DIR / params['model']
    tokenizer_path: Path = RESOURCES_DIR / params["tokenizer"]

    model = tf.keras.models.load_model(model_path)
    tokenizer = utilities.get_tokenizer(tokenizer_path)
    
    return model, tokenizer

# Test function for original vs. altered reviews (random word substitution)
def test_model_robustness_original_vs_altered(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    reviews = [
        ("The product arrived quickly and works perfectly.", "The product arrived quickly and dances perfectly."),
        ("Not worth the price, very disappointing quality.", "Not worth the price, very hilarious quality."),
        ("Great value for the money, highly recommend!", "Great value for the monkey, highly recommend!"),
        ("Easy to use and does the job well.", "Easy to hug and does the job well."),
        ("The packaging was damaged, but the item is fine.", "The packaging was damaged, but the tiger is fine.")
    ]

    results = []
    failures = []

    for original_review, altered_review in reviews:
        sentiment_orig, prob_orig = predict_sentiment(original_review, model, tokenizer)
        prob_orig_value = float(prob_orig)

        sentiment_alt, prob_alt = predict_sentiment(altered_review, model, tokenizer)
        prob_alt_value = float(prob_alt)

        # Check if the prediction is conserved
        is_conserved = check_prediction_conservation(prob_orig_value, prob_alt_value)
        result = {
            "Original": original_review,
            "Altered": altered_review,
            "Original Probability": prob_orig_value,
            "Altered Probability": prob_alt_value,
            "Conservation": is_conserved
        }
        results.append(result)

        # Collect failures for reporting
        if not is_conserved:
            failures.append(result)

    # Print the results
    print("\nTest Results:")
    for res in results:
        status = "Conserved" if res["Conservation"] else "Not Conserved"
        print(f"Original: {res['Original']}\n"
              f"Altered: {res['Altered']}\n"
              f"Original Probability: {res['Original Probability']:.4f}\n"
              f"Altered Probability: {res['Altered Probability']:.4f}\n"
              f"Conservation: {status}\n")

    # Raise an assertion error if there are any failures
    if failures:
        raise AssertionError(f"{len(failures)} out of {len(results)} tests failed in the Random Word Substitution robustness tests:\n" +
                             "\n".join([f"Original: {f['Original']} | Altered: {f['Altered']} | "
                                        f"Orig Prob: {f['Original Probability']:.4f}, "
                                        f"Altered Prob: {f['Altered Probability']:.4f}" for f in failures]))

if __name__ == "__main__":
    pytest.main()
