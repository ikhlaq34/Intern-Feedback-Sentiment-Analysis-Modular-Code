"""
features.py
Text vectorization and feature helpers.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List
from scipy.sparse import csr_matrix


def build_tfidf(corpus: List[str], max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[TfidfVectorizer, csr_matrix]:
    """Fit a TF-IDF vectorizer on the corpus and return (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


def transform_tfidf(vectorizer: TfidfVectorizer, texts: List[str]) -> csr_matrix:
    """Transform new texts using an existing vectorizer."""
    return vectorizer.transform(texts)


if __name__ == '__main__':
    sample_corpus = ["i love the mentorship", "workload was too much"]
    v, X = build_tfidf(sample_corpus, max_features=50)
    print(X.shape)