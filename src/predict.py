import pandas as pd
from sklearn.base import BaseEstimator
from typing import Dict
from preprocessing import preprocess_text
from inference import extract_entities
from joblib import load
from feature_engineering import engineer_features

# Process a single ticket: predict issue & urgency, extract entities
def process_ticket(ticket_text: str,
                   issue_model: BaseEstimator,
                   urgency_model: BaseEstimator,
                   product_list: list) -> Dict:
    """
    Process a single ticket and return prediction and extracted entities.
    """
    # Preprocess and feature engineering
    clean_text = preprocess_text(ticket_text)
    df = pd.DataFrame({
        'ticket_text': [ticket_text],
        'clean_text': [clean_text]
    })
    df = engineer_features(df)
    features = df[['clean_text', 'ticket_length', 'sentiment_score', 'avg_word_length', 'has_question', 'has_exclamation']]

    # Model predictions
    issue_pred = issue_model.predict(features)[0]
    urgency_pred = urgency_model.predict(features)[0]

    # Extract relevant entities
    entities = extract_entities(ticket_text, product_list=product_list)

    return {
        "predicted_issue_type": issue_pred,
        "predicted_urgency_level": urgency_pred,
        "extracted_entities": entities
    }

# Example run
if __name__ == "__main__":
    issue_model = load('models/issue_model.pkl')
    urgency_model = load('models/urgency_model.pkl')

    ticket = "My mobile screen is broken and it’s not charging since 3rd March."
    product_list = [
        'SmartWatch V2', 'UltraClean Vacuum', 'SoundWave 300', 'PhotoSnap Cam',
        'Vision LED TV', 'EcoBreeze AC', 'RoboChef Blender', 'FitRun Treadmill',
        'PowerMax Battery', 'ProTab X1'
    ]

    result = process_ticket(ticket, issue_model, urgency_model, product_list)
    print(result)
