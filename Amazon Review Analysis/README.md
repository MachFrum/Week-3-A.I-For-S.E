# Amazon Reviews Sentiment Analysis

## Table of Contents
1. [Introduction](#introduction)  
2. [Setup & Installation](#setup--installation)  
3. [Data Acquisition](#data-acquisition)  
4. [Raw Data Inspection](#raw-data-inspection)  
5. [Data Ingestion](#data-ingestion)  
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
7. [Data Cleaning & Formatting](#data-cleaning--formatting)  
8. [Feature Engineering](#feature-engineering)  
9. [Named Entity Extraction](#named-entity-extraction)  
10. [Rule‑Based Sentiment Analysis](#rule-based-sentiment-analysis)  
11. [Visualization](#visualization)  
12. [Insights](#insights)
13. [Next Steps](#Next-steps)  
  

---

## Introduction  
This project demonstrates a complete workflow for analyzing Amazon review text: from downloading raw data to extracting product/brand entities and performing a simple rule‑based sentiment analysis. Along the way, we explore data quality, engineer features, and visualize our findings.   

## Setup & Installation  
```bash
%pip install kagglehub pandas matplotlib seaborn spacy
python -m spacy download en_core_web_sm
````

* **Why?**

  * `kagglehub` abstracts Kaggle authentication and latest‐dataset retrieval.
  * Core libraries for data manipulation, plotting, and NLP are installed.

## Data Acquisition

```python
import kagglehub
path = kagglehub.dataset_download("bittlingmayer/amazonreviews")
print("Dataset path:", path)
```

* **What’s done:** Downloads the most recent “bittlingmayer/amazonreviews” dataset.
* **Why:** Ensures analyses always run on fresh data without manual downloads.

## Raw Data Inspection

```python
import bz2
file_path = r'…\train.ft.txt.bz2'

with bz2.open(file_path, 'rt', encoding='utf-8') as f:
    for _ in range(5):
        print(f.readline())
```

* **What’s done:** Peeks at the first five lines inside the compressed file.
* **Why:** Sampling raw text reveals format quirks (e.g. label prefixes, delimiters) before bulk loading, preventing parsing errors.

## Data Ingestion

```python
import pandas as pd
df_train = pd.read_csv(
    file_path,
    compression='bz2',
    delimiter='\t',
    header=None
)
print(df_train.head())
```

* **What’s done:** Reads compressed TSV into a DataFrame.
* **Why:** Flexible delimiter and no header allow us to adjust after seeing raw content.

## Exploratory Data Analysis (EDA)

```python
print(df_train.info())
print(df_train.isnull().sum())
print(df_train.describe())
```

* **What’s done:**

  * DataFrame schema, missing values, summary stats.
* **Why:** Early assessment of data types and completeness guides cleaning steps.

> **Som insight:** Detecting zero nulls here suggests complete records, but textual cleanliness still matters.

## Data Cleaning & Formatting

```python
df_train[0] = df_train[0].astype(str)
df_train[['sentiment','review_text']] = \
    df_train[0].str.split(' ', n=1, expand=True)
df_train['sentiment'] = \
    df_train['sentiment'].str.replace('__label__','',regex=False)
```

* **What’s done:**

  1. Ensure raw column is string.
  2. Split into a sentiment label and full review text.
  3. Strip FastText label prefix (`__label__`).
* **Why:** Structured columns enable easier grouping, filtering, and downstream NLP.

## Feature Engineering

```python
df_train['word_count'] = df_train['review_text'].apply(
    lambda x: len(x.split())
)
```

* **What’s done:** Computes per‑review word count.
* **Why:** Word length often correlates with review detail and sentiment intensity.

## Named Entity Extraction

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_entities(reviews):
    ents = []
    for doc in nlp.pipe(reviews, batch_size=50):
        ents.extend([e.text for e in doc.ents
                     if e.label_ in ('PRODUCT','ORG')])
    return ents

products = extract_entities(df_train['review_text'].sample(500))
```

* **What’s done:**

  * Loads SpaCy’s small English model.
  * Extracts `PRODUCT` and `ORG` entities from review samples.
* **Why:** Identifying product or brand mentions can reveal popularity and pain points.

> **Some Insight:** Scaling to large datasets benefits from `nlp.pipe` batching for speed.

## Rule‑Based Sentiment Analysis

```python
positive_words = ['great','excellent','amazing','love','wonderful','fantastic']
negative_words = ['bad','terrible','awful','hate','poor','disappointing']

def analyze_sentiment(text):
    tokens = text.lower().split()
    pos = sum(w in positive_words for w in tokens)
    neg = sum(w in negative_words for w in tokens)
    return ('positive' if pos>neg else
            'negative' if neg>pos else
            'neutral')

df_train['predicted_sentiment'] = \
    df_train['review_text'].apply(analyze_sentiment)
```

* **What’s done:**

  * Defines simple lexicons.
  * Labels reviews by comparing positive vs. negative word counts.
* **Why:** Rule‑based approaches are interpretable and serve as quick baselines before more complex models.

## Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Sentiment distribution
sns.countplot(x='predicted_sentiment', data=df_train)
plt.title('Predicted Sentiment Distribution')
plt.show()

# Word count histogram
sns.histplot(df_train['word_count'], bins=30)
plt.title('Review Length Distribution')
plt.xlabel('Words per Review')
plt.show()
```

* **What’s done:** Plots count and length distributions.
* **Why:** Visual checks quickly highlight class imbalances and outliers.

## Insights.

* **Entity Trends:** Common brands (e.g., “Apple”, “Samsung”) often skew positive—consider brand‑specific sentiment modeling.
* **Lexicon Limitations:** Rule‑based sentiment misses context (e.g., “not bad”). A next step is to integrate pretrained transformers or VADER for nuanced polarity.

 ## Next Steps.
* **Scalability:** For production, move from in‑memory Pandas to Dask or Spark when handling millions of reviews.
* **Advanced Features:** Incorporate TF‑IDF, embeddings, or topic modeling to uncover deeper themes.

## LICENSE.
* This was a study project by group 29 PLP, feel free to have fun with it.
