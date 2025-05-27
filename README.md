# Project Overview

This project implements a multi-task NLP pipeline for analyzing customer support tickets. It processes raw ticket text to:

- Predict **Issue Type** (multi-class classification)
- Predict **Urgency Level** (Low, Medium, High)
- Extract key entities: product names, dates, and complaint keywords (rule-based NLP)

The pipeline covers data preprocessing, feature engineering, model training, and provides an interactive Gradio web interface.

---

## Data Preparation

**Input:** Excel file `ai_dev_assignment_tickets_complex_1000` with columns: `ticket_id`, `ticket_text`, `issue_type`, `urgency_level`, `product`.

### Preprocessing (`preprocess.py`)
- **Cleans and normalizes** ticket text (lowercasing, removing special characters, lemmatization with SpaCy, stopword removal).
- **Handles missing data:** Drops rows missing `ticket_id`/`ticket_text`; fills missing labels with `"unknown"`.
- **Adds** a `clean_text` column to the DataFrame.

**Usage:** Run the script or import `load_and_preprocess_data()`.

### Product List Extraction (`prepare_product_list.py`)
- Extracts unique product names from the dataset.
- Saves the product list as `models/product_list.pkl` for entity extraction.

---

## Feature Engineering (`feature_engineering.py`)

Creates features from cleaned text:

- **Textual Features:**
    - `ticket_length`: Word count
    - `avg_word_length`: Average word length
    - `sentiment_score`: Polarity from TextBlob
    - `has_question`: Contains question mark
    - `has_exclamation`: Contains exclamation mark

- **Vectorized Features:**
    - Bag-of-Words (BoW)
    - TF-IDF (unigrams & bigrams, max 5000 features)

**Functions:**
- `engineer_features(df)`: Adds features to DataFrame
- `get_vectorizers(df, max_features=5000)`: Fits vectorizers
- `transform_with_vectorizers(df, ...)`: Transforms new data

---

## Modeling (`model.py`)

Trains two independent classifiers:

- **Issue Type:** Logistic Regression (multi-class)
- **Urgency Level:** Random Forest (multi-class)

**Approach:**
- Uses TF-IDF vectors + engineered features (standardized)
- Pipelines for preprocessing and modeling
- 80/20 train-test split, random seed fixed

**Functions:**
- `build_and_train_models(...)`: Trains, evaluates, saves models (`models/issue_model.pkl`, `models/urgency_model.pkl`)
- `plot_conf_matrix(...)`: Plots confusion matrices

---

## Entity Extraction (`inference.py`)

Extracts entities from ticket text using regex, fuzzy matching, and SpaCy:

- **Product Names:** Fuzzy match against product list
- **Dates:** Regex for absolute/relative formats
- **Complaint Keywords:** Predefined list, lemmatized matching

**Functions:**
- `extract_entities(text, product_list, threshold=80)`: Returns dict with `products`, `dates`, `complaints`
- `extract_entities_for_dataframe(df, product_column='product')`: Applies extraction row-wise

**Example Output:**
```json
{
    "products": ["ProductA"],
    "dates": ["2025-05-20"],
    "complaints": ["broken", "error"]
}
```

---

## Integration & Interface

- `process_ticket(ticket_text)`: Cleans text, extracts features, predicts labels, extracts entities, returns results as JSON/dict.
- **Gradio App:** User inputs ticket text, receives predictions and extracted entities. Supports batch processing.

---

## Evaluation & Limitations

- Models achieve approximately [insert accuracy/F1] on test data.
- **Limitations:** Rule-based entity extraction may miss entities; classical ML models may not capture complex context; possible dataset imbalance.
- **Future Work:** Explore deep learning and advanced NER.

---

## How to Run

1. **Install dependencies:**
     ```bash
     pip install -r requirements.txt
     ```
2. **Train/test models:** Run main script or Jupyter notebook.
3. **Launch Gradio app:**
     ```bash
     python app.py
     ```
     Open the provided URL.

---

## File Structure

```
├── data/            # Raw and processed data
├── notebooks/       # EDA and model training notebooks
├── models/          # Saved models and product list
├── src/             # Code modules
├── app.py           # Gradio interface
├── requirements.txt # Dependencies
└── README.md        # Documentation
```

---

## Acknowledgments

- NLTK (NLP tools)
- Scikit-learn (ML models)
- Gradio (web interface)
- SpaCy, TextBlob, FuzzyWuzzy

