"""
Sarcasm Detection Tests
"""

from pathlib import Path
import sys
import typing
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

def predict_sentiment(text: str, model: tf.keras.Model, tokenizer) -> typing.Tuple[str, float]:
    """
    Predict sentiment for a given text using a pre-trained model and tokenizer.
    """
    sequence = tokenizer.texts_to_sequences([text])  # Convert text to sequence
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, padding='post', maxlen=250)
    prediction = model.predict(padded_sequence)[0]
    sentiment_label = 'Negative' if prediction < 0.5 else 'Positive'
    return sentiment_label, prediction

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

# Test function for original vs. sarcastic reviews
def test_model_sarcasm_detection(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    sarcasm_reviews = [
        ("My order took three damn weeks to arrive, which is annoying.", "It took three damn weeks for my order to arrive, which is so convenient."),
        ("I hate waiting in long periods of time for my product to arrive!", "Wow, I love waiting in long periods of time for my product to arrive!"),
        ("What a way to ruin my Monday morning, my CD reader failed!", "My CD reader failed! This is exactly what I needed to ruin my Monday morning!"),
        ("Absolutely annoyed by this awful service!", "Absolutely thrilled about this awful service!"),
        ("I'm so angry, my phone battery died during ruining the important call.", "I'm so glad my phone battery died and ruined the important call.")
    ]

    results = []
    failures = []

    for original_review, sarcastic_review in sarcasm_reviews:
        sentiment_orig, prob_orig = predict_sentiment(original_review, model, tokenizer)
        prob_orig_value = float(prob_orig)

        sentiment_sarc, prob_sarc = predict_sentiment(sarcastic_review, model, tokenizer)
        prob_sarc_value = float(prob_sarc)

        # Check if the prediction is conserved
        is_conserved = check_prediction_conservation(prob_orig_value, prob_sarc_value)
        result = {
            "Original": original_review,
            "Sarcastic": sarcastic_review,
            "Original Probability": prob_orig_value,
            "Sarcastic Probability": prob_sarc_value,
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
              f"Sarcastic: {res['Sarcastic']}\n"
              f"Original Probability: {res['Original Probability']:.4f}\n"
              f"Sarcastic Probability: {res['Sarcastic Probability']:.4f}\n"
              f"Conservation: {status}\n")

    # Raise an assertion error if there are any failures
    if failures:
        raise AssertionError(f"{len(failures)} out of {len(results)} tests failed in the Sarcasm robustness tests:\n" +
                             "\n".join([f"Original: {f['Original']} | Sarcastic: {f['Sarcastic']} | "
                                        f"Orig Prob: {f['Original Probability']:.4f}, "
                                        f"Sarcastic Prob: {f['Sarcastic Probability']:.4f}" for f in failures]))

if __name__ == "__main__":
    pytest.main()
