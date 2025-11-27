"""
preprocessing.py
Text cleaning and simple feature helpers.
"""
import re
from typing import List


def clean_text(text: str) -> str:
    """Lowercase, remove non-alphabetic characters, and collapse whitespace."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r"[^a-z\s]", ' ', text)
    text = ' '.join(text.split())
    return text


def tokenize(text: str) -> List[str]:
    """Very small tokenizer splitting on whitespace. Kept for extensibility."""
    return clean_text(text).split()


if __name__ == '__main__':
    example = "The mentorship program was GREAT!!!"
    print(clean_text(example))
    print(tokenize(example))