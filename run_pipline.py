import pandas as pd
import numpy as np
from data_generation import generate_feedback
from processing import clean_text
from features import build_tfidf, transform_tfidf
from modeling import train_logistic_regression, evaluate_classifier, predict_with_confidence, save_model, load_model
from analysis_tool import extract_common_words, plot_sentiment_distribution, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_dataframe(raw_data):
    df = pd.DataFrame(raw_data)
    df['cleaned_text'] = df['feedback_text'].apply(clean_text)
    return df


def auto_label_textblob_demo(df):
    def heur_label(txt):
        txt = txt.lower()
        if any(w in txt for w in ['excellent', 'great', 'amazing', 'outstanding', 'wonderful', 'fantastic']):
            return 'positive'
        if any(w in txt for w in ['poor', 'terrible', 'awful', 'disappointing', 'inadequate', 'frustrating']):
            return 'negative'
        return 'neutral'
    
    df['sentiment'] = df['feedback_text'].apply(heur_label)
    return df


def main(seed: int = 42):
    np.random.seed(seed)

    # 1) Data generation
    raw = generate_feedback(200, seed=seed)
    df = prepare_dataframe(raw)
    print(f"Generated {len(df)} samples")

    # 2) Label (auto)
    df = auto_label_textblob_demo(df)
    print('Label distribution:')
    print(df['sentiment'].value_counts())

    # 3) TF-IDF
    vectorizer, X = build_tfidf(df['cleaned_text'].tolist(), max_features=1000)

    # 4) Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # 6) Train
    model = train_logistic_regression(X_train, y_train)

    # 7) Evaluate
    results = evaluate_classifier(model, X_test, y_test, target_names=le.classes_)
    print(f"Accuracy: {results['accuracy']:.4f}")

    # 8) Visualize
    plot_sentiment_distribution(df['sentiment'].tolist(), label_names=list(le.classes_))
    plot_confusion_matrix(results['confusion_matrix'], labels=list(le.classes_))


if __name__ == '__main__':
    main()