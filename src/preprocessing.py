import pandas as pd
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lemmatize and remove stopwords/punctuation/whitespace
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    
    return ' '.join(tokens)

def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(subset=['ticket_id', 'ticket_text'])
    
    print("Missing values before filling:")
    print(df[['issue_type', 'urgency_level', 'product']].isnull().sum())
    
    for col in ['issue_type', 'urgency_level', 'product']:
        df[col] = df[col].fillna('unknown')
    
    df['clean_text'] = df['ticket_text'].apply(preprocess_text)
    return df

if __name__ == "__main__":
    df = load_and_preprocess_data("data/raw/ai_dev_assignment_tickets_complex_1000.xls")
    print(df[['ticket_id', 'clean_text']].head())
