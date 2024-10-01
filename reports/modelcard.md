# Model Card for GPT-2 Amazon Sentiment Classifier (V1.0)

## Model Summary

This model is a fine-tuned version of GPT-2 for sentiment classification of Amazon reviews. It was trained on a dataset of Amazon product reviews to classify text as having positive or negative sentiment.

## Model Details

### Model Description

The GPT-2 Amazon Sentiment Classifier is built on the GPT-2 architecture, fine-tuned specifically for sentiment analysis of product reviews. The model is capable of identifying the overall sentiment expressed in an Amazon product review and categorizing it into either positive or negative. It leverages the powerful language understanding abilities of GPT-2 to perform text classification tasks, focusing on customer feedback in e-commerce.

- **Developed by:** ashok2216  
- **Funded by:** [More Information Needed]
- **Shared by:** ashok2216  
- **Model type:** GPT-2 fine-tuned for sentiment classification  
- **Language(s) (NLP):** English  
- **License:** MIT License
- **Finetuned from model:** GPT-2  

### Model Sources

- **Repository:** [GPT-2 Amazon Sentiment Classifier V1.0 on Hugging Face](https://huggingface.co/ashok2216/gpt2-amazon-sentiment-classifier-V1.0)
- **Paper:** [More Information Needed]
- **Demo:** [More Information Needed]

## Uses

### Direct Use

This model can be used directly for sentiment analysis of product reviews on e-commerce platforms. It can be integrated into applications to gauge customer feedback automatically, classifying reviews into positive or negative sentiment categories.

### Downstream Use

### Out-of-Scope Use

## Bias, Risks, and Limitations

The model, trained primarily on Amazon reviews, may contain biases reflecting the specific product categories and customer demographics represented in the dataset. There is a risk that the model may underperform when faced with reviews from other e-commerce platforms or industries. Additionally, certain sentiments could be misclassified, particularly if the review contains sarcasm, irony, or complex language.

### Recommendations

## How to Get Started with the Model

```python
from transformers import pipeline

model_name = "ashok2216/gpt2-amazon-sentiment-classifier-V1.0"
classifier = pipeline("sentiment-analysis", model=model_name)

result = classifier("This product is excellent and I love using it!")
print(result)
```

## Training Details

### Training Data

The model was trained on a dataset of Amazon product reviews. This dataset contains customer reviews spanning multiple product categories and consists of text that expresses customer sentiment in response to their purchasing experience. It also cotnains other fields and metadata.

### Training Procedure

#### Preprocessing

#### Training Hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2
  
#### Speeds, Sizes, Times

[More Information Needed]

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model's performance was evaluated on a held-out test set of Amazon reviews that were not included in the training data.

#### Factors

The evaluation focused on the correct classification of sentiment across different types of product categories and varying lengths of reviews.

#### Metrics

Standard classification metrics, including accuracy and F1 score, are used. These metrics assess how well the model differentiates between positive and negative sentiment categories.

### Results

The model achieves the following results on the evaluation set:
- Loss: 0.0320
- Accuracy: 0.9680
- F1: 0.9680

#### Summary

[More Information Needed]

## Model Examination

[More Information Needed]

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications

### Model Architecture and Objective

The model is based on the GPT-2 architecture, fine-tuned for sentiment analysis using Amazon review data.

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

The model uses the Hugging Face Transformers library and is compatible with Python environments.

The following framework versions were used:
- Transformers 4.39.3
- Pytorch 2.2.1+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2

## Citation

**BibTeX:**

```bibtex
@misc{ashok_gpt2_amazon_sentiment,
  author = {ashok2216},
  title = {GPT-2 Amazon Sentiment Classifier V1.0},
  year = {2023},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/ashok2216/gpt2-amazon-sentiment-classifier-V1.0}},
}
```

**APA:**

[More Information Needed]

## Glossary

[More Information Needed]

## More Information

[More Information Needed]

## Model Card Authors

Benji33, EnricDataS, lluc-palou

## Model Card Contact

benedikt.blank@estudiantat.upc.edu
enric.millan.iglesias@estudiantat.upc.edu
lluc.palou@estudiantat.upc.edu
