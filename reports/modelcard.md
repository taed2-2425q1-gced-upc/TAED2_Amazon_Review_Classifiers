---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
---

# Model Card for hassansattar/sentimental-customer-review

This model performs sentiment analysis on Amazon reviews, classifying them as either positive or negative.

## Model Details

### Model Description

The model processes customer reviews and predicts their sentiment as either positive or negative. It uses a transformer architecture pre-trained GloVe embedding matrix and bidirectional LSTM structures.

- **Developed by:** Hassan Sattar
- **Model type:** LSTM
- **Language(s) (NLP):** English
- **License:** Unlicense

### Model Sources 

- **Repository:** [Hugging Face](https://huggingface.co/hassansattar/sentimental-customer-review/tree/main)

## Uses

### Direct Use

This model can be used directly for analyzing the sentiment of text reviews. It is particularly suited for customer review data but may generalize to other forms of sentiment analysis.

### Out-of-Scope Use

The model is not designed for tasks outside of binary sentiment classification (positive vs. negative) and may not perform well with highly subjective, complex emotions.

## Bias, Risks, and Limitations

The model is trained on a specific dataset (Amazon reviews), which may introduce bias depending on the type of reviews and sentiments present in that data. It may not generalize well to datasets with different linguistic styles or cultural nuances.

### Recommendations

Users should be cautious when applying the model to domains that diverge significantly from the Amazon reviews dataset. More rigorous evaluations are necessary when deploying the model in such environments.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from tensorflow.keras.models import load_model

# Load the model
model = load_model('sentiment_model.h5')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(texts, tokenizer, max_length=250):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

def predict_sentiment(text, model, tokenizer):
    preprocessed_text = preprocess_text(text, tokenizer)
    prediction = model.predict(preprocessed_text)[0]
    print("Prediction:", prediction)
    sentiment_label = 'Negative' if prediction < 0.5 else 'Positive'
    return sentiment_label

# Example review
new_review = "very bad product"

# Predict sentiment
sentiment = predict_sentiment(new_review, model, tokenizer)
print(f"Review: {new_review}\nSentiment: {sentiment}\n")
```

## Training Details

### Training Data

The training data is based on a  dataset of Amazon reviews, where each review is labeled as either positive or negative. 

#### Preprocessing

Data preprocessing involves tokenizing the text into sequences and padding them for uniform length.

#### Training Hyperparameters

The model structure is the following:

```python
model = Sequential([
    Embedding(10000, embedding_dim, embeddings_initializer=Constant(embedding_matrix), 
              input_length=250, trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(128)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

The optimizer is adam and the loss is the binary cross-entropy loss.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Total params: 1,628,993

Trainable params: 628,993

Non-trainable params: 1,000,000

The model is trained for 5 epochs, with a batch size of 64.

## Evaluation

### Testing Data, Factors & Metrics


#### Testing Data

The testing data is the validation set split from the same Amazon reviews dataset.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

The primary evaluation metric used is accuracy, alongside loss metrics to track performance during training.

### Results

The model achieved high accuracy on the validation dataset, demonstrating strong performance in binary sentiment classification. The accuracy on validation was 0.9132547974586487. Validation loss was  0.23057065904140472.

#### Summary

[More Information Needed]


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]


### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

-TensorFlow 2.10.0

## Model Card Authors

Benji33, EnricDataS, lluc-palou

## Model Card Contact

benedikt.blank@estudiantat.upc.edu
enric.millan.iglesias@estudiantat.upc.edu
lluc.palou@estudiantat.upc.edu
