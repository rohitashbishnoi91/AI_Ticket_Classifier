import joblib
import pandas as pd
from preprocessing import preprocess_text
from feature_engineering import engineer_features
from inference import extract_entities

class TicketClassifier:
    """
    Classifies support tickets and extracts entities.
    """
    def __init__(self,
                 issue_model_path='models/issue_model.pkl',
                 urgency_model_path='models/urgency_model.pkl',
                 product_list_path='models/product_list.pkl'):
        
        # Load pre-trained models and product list
        self.issue_model = joblib.load(issue_model_path)
        self.urgency_model = joblib.load(urgency_model_path)
        self.product_list = joblib.load(product_list_path)

    def predict(self, text):
        # Preprocess text and create DataFrame
        clean_text = preprocess_text(text)
        df = pd.DataFrame({
            'ticket_text': [text],
            'clean_text': [clean_text]
        })

        # Generate features
        df = engineer_features(df)

        # Select required features for prediction
        features = df[['clean_text', 'ticket_length', 'sentiment_score', 'avg_word_length', 'has_question', 'has_exclamation']]

        # Make predictions
        issue_pred = self.issue_model.predict(features)[0]
        urgency_pred = self.urgency_model.predict(features)[0]

        # Extract entities from original text
        entities = extract_entities(text, product_list=self.product_list)

        return {
            "issue_type": issue_pred,
            "urgency_level": urgency_pred,
            "entities": entities
        }
