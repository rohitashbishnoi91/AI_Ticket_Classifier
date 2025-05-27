import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob

def engineer_features(df):
    """
    Enhances the input DataFrame with additional engineered features for text analysis.
    Parameters:
        df (pd.DataFrame): Must contain 'clean_text' and 'ticket_text' columns.
    Returns:
        pd.DataFrame: DataFrame with added feature columns.
    """
    
    # Total number of words in the clean text
    df['ticket_length'] = df['clean_text'].apply(lambda x: len(x.split()))
    
    # Average length of each word
    df['avg_word_length'] = df['clean_text'].apply(
        lambda x: sum(len(w) for w in x.split()) / (len(x.split()) + 1e-5)
    )
    
    # Sentiment polarity score using TextBlob (-1 to 1)
    df['sentiment_score'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Binary indicator if a question mark is present
    df['has_question'] = df['ticket_text'].str.contains('\\?', regex=True).astype(int)
    
    # Binary indicator if an exclamation mark is present
    df['has_exclamation'] = df['ticket_text'].str.contains('!', regex=True).astype(int)
    
    return df


def get_vectorizers(df, max_features=5000):
    """
    Initializes and fits BoW and TF-IDF vectorizers on the clean text.
    Returns:
        Feature matrices and fitted vectorizers.
    """
    bow_vectorizer = CountVectorizer(max_features=max_features)
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

    # Fit and transform text into vectors
    X_bow = bow_vectorizer.fit_transform(df['clean_text'])
    X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])

    return X_bow, X_tfidf, bow_vectorizer, tfidf_vectorizer


def transform_with_vectorizers(df, bow_vectorizer, tfidf_vectorizer):
    """
    Transforms new data using previously fitted vectorizers.
    Returns:
        Transformed BoW and TF-IDF feature matrices.
    """
    X_bow = bow_vectorizer.transform(df['clean_text'])
    X_tfidf = tfidf_vectorizer.transform(df['clean_text'])
    return X_bow, X_tfidf
