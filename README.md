# Project Overview

This project implements a multi-task NLP pipeline for analyzing customer support tickets. It processes raw ticket text to:

* Predict **Issue Type** (multi-class classification)
* Predict **Urgency Level** (Low, Medium, High)
* Extract key entities: product names, dates, and complaint keywords (rule-based NLP)

The pipeline covers data preprocessing, feature engineering, model training, and provides an interactive Gradio web interface.

---

## Data Preparation

**Input:** Excel file `ai_dev_assignment_tickets_complex_1000` with columns: `ticket_id`, `ticket_text`, `issue_type`, `urgency_level`, `product`.

### Preprocessing (`preprocess.py`)

* **Cleans and normalizes** ticket text (lowercasing, removing special characters, lemmatization with SpaCy, stopword removal).
* **Handles missing data:** Drops rows missing `ticket_id`/`ticket_text`; fills missing labels with `"unknown"`.
* **Adds** a `clean_text` column to the DataFrame.

**Usage:** Run the script or import `load_and_preprocess_data()`.

### Product List Extraction (`prepare_product_list.py`)

* Extracts unique product names from the dataset.
* Saves the product list as `models/product_list.pkl` for entity extraction.

---

## Feature Engineering (`feature_engineering.py`)

Creates features from cleaned text:

* **Textual Features:**

  * `ticket_length`: Word count
  * `avg_word_length`: Average word length
  * `sentiment_score`: Polarity from TextBlob
  * `has_question`: Contains question mark
  * `has_exclamation`: Contains exclamation mark

* **Vectorized Features:**

  * Bag-of-Words (BoW)
  * TF-IDF (unigrams & bigrams, max 5000 features)

**Functions:**

* `engineer_features(df)`: Adds features to DataFrame
* `get_vectorizers(df, max_features=5000)`: Fits vectorizers
* `transform_with_vectorizers(df, ...)`: Transforms new data

---

## Modeling (`model.py`)

Trains two independent classifiers:

* **Issue Type:** Logistic Regression (multi-class)
* **Urgency Level:** Random Forest (multi-class)

**Approach:**

* Uses TF-IDF vectors + engineered features (standardized)
* Pipelines for preprocessing and modeling
* 80/20 train-test split, random seed fixed

**Functions:**

* `build_and_train_models(...)`: Trains, evaluates, saves models (`models/issue_model.pkl`, `models/urgency_model.pkl`)
* `plot_conf_matrix(...)`: Plots confusion matrices

---

## Entity Extraction (`inference.py`)

Extracts entities from ticket text using regex, fuzzy matching, and SpaCy:

* **Product Names:** Fuzzy match against product list
* **Dates:** Regex for absolute/relative formats
* **Complaint Keywords:** Predefined list, lemmatized matching

**Functions:**

* `extract_entities(text, product_list, threshold=80)`: Returns dict with `products`, `dates`, `complaints`
* `extract_entities_for_dataframe(df, product_column='product')`: Applies extraction row-wise

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

* `process_ticket(ticket_text)`: Cleans text, extracts features, predicts labels, extracts entities, returns results as JSON/dict.
* **Gradio App:** User inputs ticket text, receives predictions and extracted entities. Supports batch processing.

---

## Evaluation & Limitations

###  **Missing Value Summary Before Filling**

```plaintext
issue_type       74
urgency_level    49
product           0
```

###  **Issue Type Classifier Report**

```plaintext
                    precision    recall  f1-score   support

    Account Access       0.93      0.96      0.94        26
   Billing Problem       0.96      1.00      0.98        24
   General Inquiry       0.91      1.00      0.96        32
Installation Issue       0.91      1.00      0.95        29
     Late Delivery       0.96      1.00      0.98        24
    Product Defect       0.81      1.00      0.90        13
        Wrong Item       0.89      1.00      0.94        25
           unknown       0.00      0.00      0.00        16

          accuracy                           0.91       189
         macro avg       0.80      0.87      0.83       189
      weighted avg       0.84      0.91      0.87       189
```

### 🕐 **Urgency Level Classifier Report**

```plaintext
              precision    recall  f1-score   support

        High       0.36      0.27      0.31        73
         Low       0.22      0.28      0.25        53
      Medium       0.32      0.34      0.33        58
     unknown       0.00      0.00      0.00         5

    accuracy                           0.29       189
   macro avg       0.23      0.23      0.22       189
weighted avg       0.30      0.29      0.29       189
```

---

###  **Confusion Matrices**

#### **Issue Type**
![Screenshot (237)](https://github.com/user-attachments/assets/1bfc9868-af5f-4451-a959-e1fdc327aa30)

#### **Urgency Level**


![Screenshot (238)](https://github.com/user-attachments/assets/4b114193-18da-4d56-bc78-912961a94da7)



---

### Limitations

* Rule-based **entity extraction** may miss nuanced expressions (e.g., slang or typos).
* Classical ML models lack deep contextual understanding.
* **Class imbalance** in `urgency_level` affects performance.
* "unknown" labels likely hurt classification accuracy due to overlap and ambiguity.

---

###  Future Work

* Switch to transformer-based models (e.g., BERT) for classification and NER.
* Add model explainability with SHAP or LIME.
* Improve date and complaint keyword recognition using dependency parsing or NER fine-tuning.

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

* NLTK (NLP tools)
* Scikit-learn (ML models)
* Gradio (web interface)
* SpaCy, TextBlob, FuzzyWuzzy
