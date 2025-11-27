"""
analysis_tools.py
Helpers for extracting themes, plotting, and summarizing results.
"""
from collections import Counter
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def extract_common_words(texts: List[str], top_n: int = 10, stopwords: List[str] = None) -> List[str]:
    stop = set(stopwords or [])
    all_words = []
    for t in texts:
        words = t.lower().split()
        all_words.extend([w for w in words if w not in stop])
    
    most = Counter(all_words).most_common(top_n)
    return [w for w, _ in most]


def plot_sentiment_distribution(labels, label_names=None):
    counts = Counter(labels)
    names = label_names if label_names else sorted(counts.keys())
    values = [counts[n] for n in names]
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=names, y=values)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_sentiment_distribution(['positive']*5 + ['negative']*2 + ['neutral']*1)