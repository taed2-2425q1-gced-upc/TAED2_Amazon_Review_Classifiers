# Dataset Card for Amazon Reviews 2023

The Amazon Reviews 2023 dataset, collected by McAuley Lab, contains over 571 million reviews from May 1996 to September 2023, covering a wide variety of product categories. It includes user reviews, item metadata, and user-item interactions, making it suitable for tasks such as sentiment analysis, text classification, and recommendation systems (RecSys).

## Dataset Details
### Dataset Description

The Amazon Reviews 2023 dataset includes rich features such as user reviews (ratings, text, helpfulness votes), item metadata (descriptions, price, images), and user-item links. This dataset is valuable for tasks like sentiment analysis, text classification, and recommendation systems, and is widely used in natural language processing (NLP) research. Standard data splits are provided to facilitate RecSys benchmarking.

- **Curated by:** McAuley Lab
- **Funded by [optional]:** University of California, San Diego
- **Shared by [optional]:** McAuley Lab
- **Language(s) (NLP):** English
- **License:** To be confirmed

### Dataset Sources

- **Repository:** [Amazon Reviews 2023 dataset](https://nijianmo.github.io/amazon/index.html)
- **Paper [optional]:** [Amazon Product Review Dataset](https://cseweb.ucsd.edu/~jmcauley/pdfs/www15.pdf)
- **Demo [optional]:** Not available

## Uses
### Direct Use

The dataset can be used directly for sentiment analysis, product recommendation systems, text classification, and as a benchmark for large-scale recommendation systems (RecSys).

### Out-of-Scope Use

The dataset should not be used for tasks that require real-time data, as it only includes data until September 2023. Additionally, the data should not be used for identifying specific individuals or making inferences about private data, as it only contains publicly available product reviews.

## Dataset Structure

The dataset includes the following fields:

- **Review Text:** The written review from users.
- **Rating:** The numeric rating given by the user (1-5).
- **Helpfulness Votes:** The number of votes indicating how helpful the review was.
- **Item Metadata:** Information such as product title, price, category, and more.
- **User-Item Links:** Links between users and the products they reviewed.

Standard train, validation, and test splits are provided to facilitate benchmarking.

## Dataset Creation
### Curation Rationale

The dataset was created to enable research in sentiment analysis, text classification, and recommendation systems at scale. It is useful for studying user behaviors, product preferences, and improving product recommendation models.

### Source Data

The data was sourced from publicly available Amazon reviews between May 1996 and September 2023. The dataset includes over 571 million reviews, with cleaned and preprocessed text for easier analysis.

### Annotations

The data was crowdsourced, relying on Amazon's platform where users submit their reviews. No additional annotations were performed beyond the available metadata.

#### Who are the annotators?

The annotators are Amazon users who voluntarily submitted their reviews on the platform.

#### Personal and Sensitive Information

The dataset does not contain any personal or sensitive information. It only includes publicly available product reviews and metadata.

## Bias, Risks, and Limitations

The dataset may contain biases, such as over-representation of certain product categories, geographical regions, or user demographics, as it depends on Amazon's customer base. Additionally, user reviews are subjective and could be influenced by factors outside of the product itself.

### Recommendations

Users should be aware of the potential biases in the dataset, such as uneven distribution of product categories and varying review frequencies over time. It is recommended to analyze the data before use to understand these biases.

## Citation

**BibTeX:**

```bibtex
@inproceedings{ni2023amazon,
  title={Amazon Reviews 2023 Dataset},
  author={McAuley Lab},
  year={2023},
  journal={University of California, San Diego}
}
```

## Dataset Card Authors

Benji33, EnricDataS, lluc-palou

## Dataset Card Contact

benedikt.blank@estudiantat.upc.edu
enric.millan.iglesias@estudiantat.upc.edu
lluc.palou@estudiantat.upc.edu