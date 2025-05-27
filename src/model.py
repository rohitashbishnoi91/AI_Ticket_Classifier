import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_and_train_models(
    df,
    text_col='clean_text',
    meta_features=['ticket_length', 'sentiment_score', 'avg_word_length', 'has_question', 'has_exclamation'],
    issue_model_path='models/issue_model.pkl',
    urgency_model_path='models/urgency_model.pkl'
):
    # Targets
    y_issue_type = df['issue_type']
    y_urgency_level = df['urgency_level']

    # Train-test split
    X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test = train_test_split(
        df[[text_col] + meta_features], y_issue_type, y_urgency_level, test_size=0.2, random_state=42
    )

    # Preprocessing
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    preprocessor = ColumnTransformer([
        ('tfidf', tfidf, text_col),
        ('meta', StandardScaler(), meta_features)
    ])

    # Pipelines
    pipeline_issue = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    pipeline_urgency = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    pipeline_issue.fit(X_train, y_issue_train)
    pipeline_urgency.fit(X_train, y_urgency_train)

    # Save models
    joblib.dump(pipeline_issue, issue_model_path)
    joblib.dump(pipeline_urgency, urgency_model_path)

    # Predict
    y_issue_pred = pipeline_issue.predict(X_test)
    y_urgency_pred = pipeline_urgency.predict(X_test)

    # Reports
    print("\n Issue Type Classifier Report:\n")
    print(classification_report(y_issue_test, y_issue_pred))
    print("\n Urgency Level Classifier Report:\n")
    print(classification_report(y_urgency_test, y_urgency_pred))

    # Confusion matrices
    plot_conf_matrix(y_issue_test, y_issue_pred, 'Confusion Matrix: Issue Type',
                     labels=sorted(y_issue_test.unique()))
    plot_conf_matrix(y_urgency_test, y_urgency_pred, 'Confusion Matrix: Urgency Level',
                     labels=['Low', 'Medium', 'High'])

    return pipeline_issue, pipeline_urgency

def plot_conf_matrix(y_true, y_pred, title, labels):
    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from preprocessing import load_and_preprocess_data
    from feature_engineering import engineer_features

    # Load and preprocess
    df = load_and_preprocess_data("data/raw/ai_dev_assignment_tickets_complex_1000.xls")
    df = engineer_features(df)

    # Train models
    build_and_train_models(
        df,
        text_col='clean_text',
        meta_features=['ticket_length', 'sentiment_score', 'avg_word_length', 'has_question', 'has_exclamation']
    )
