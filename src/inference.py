import re
from collections import defaultdict
from fuzzywuzzy import fuzz, process
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Complaint keywords (lemmatized)
complaint_keywords = [
    "broken", "error", "faulty", "delay", "late", "crash",
    "damage", "missing", "defective", "not work", "issue", "problem",
    "disconnect", "drop", "freeze", "slow", "fail", "bug"
]

# Regex patterns for absolute & relative date-like expressions
date_patterns = [
    r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',                          # 12/05/2024
    r'\b(?:\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',                            # 2024-05-12
    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}\b',  # March 3
    r'\b\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b',  # 3 March
    r'\b(today|yesterday|tomorrow|tonight|last night|every [a-z]+)\b'  # Relative
]


def extract_entities(text: str, product_list: list, threshold: int = 80) -> dict:
    """
    Extract dates, complaints, and product names from the given text.
    """
    entities = defaultdict(list)
    text_lower = text.lower()
    doc = nlp(text)

    # Extract dates using multiple patterns
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text_lower, re.IGNORECASE))
    entities['dates'] = list(set(dates))

    # Extract complaint keywords using lemmatization
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    complaints = [kw for kw in complaint_keywords if any(kw in lemma for lemma in lemmas)]
    entities['complaints'] = list(set(complaints))

    # Extract best product using fuzzy matching
    best_match = process.extractOne(text, product_list, scorer=fuzz.partial_ratio)
    if best_match and best_match[1] >= threshold:
        entities['product'] = best_match[0]

    return dict(entities)


def extract_entities_for_dataframe(df, product_column='product'):
    """
    Apply extract_entities row-wise on a dataframe.
    """
    product_list = df[product_column].dropna().unique().tolist()
    df['extracted_entities'] = df['ticket_text'].apply(lambda x: extract_entities(x, product_list))
    return df
