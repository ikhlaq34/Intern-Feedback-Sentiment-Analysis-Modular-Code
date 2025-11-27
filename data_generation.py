"""
Generates synthetic intern feedback data for experimentation.
"""
import random
from typing import List, Dict


ASPECTS = [
    'training', 'mentorship', 'work environment', 'projects', 'team collaboration',
    'feedback system', 'learning opportunities', 'work-life balance'
]


POSITIVE_ADJ = ['excellent', 'outstanding', 'amazing', 'fantastic', 'wonderful', 'great']
NEGATIVE_ADJ = ['poor', 'terrible', 'disappointing', 'inadequate', 'frustrating', 'awful']


POSITIVE_TEMPLATES = [
    "I really appreciated the {aspect} and found it {adj}.",
    "The {aspect} was {adj} and helped me grow professionally.",
    "I'm grateful for the {aspect}, it was truly {adj}.",
]


NEGATIVE_TEMPLATES = [
    "The {aspect} was {adj} and needs improvement.",
    "I found the {aspect} to be {adj} and frustrating.",
    "The {aspect} was {adj} and affected my experience.",
]


NEUTRAL_TEMPLATES = [
    "The {aspect} was acceptable but could be better.",
    "The {aspect} was average, nothing special.",
    "The {aspect} was okay but has room for improvement.",
]


def generate_feedback(n: int = 200, seed: int = 42) -> List[Dict[str, str]]:
    """Generate `n` synthetic feedback samples.

    Returns a list of dicts: {'intern_id': str, 'feedback_text': str}
    """
    random.seed(seed)
    data = []

    for i in range(1, n + 1):
        intern_id = f"INTERN_{i:03d}"
        # probabilities tuneable
        sentiment_type = random.choices(['positive', 'negative', 'neutral'], weights=[0.4, 0.3, 0.3])[0]

        if sentiment_type == 'positive':
            template = random.choice(POSITIVE_TEMPLATES)
            feedback = template.format(aspect=random.choice(ASPECTS), adj=random.choice(POSITIVE_ADJ))
        elif sentiment_type == 'negative':
            template = random.choice(NEGATIVE_TEMPLATES)
            feedback = template.format(aspect=random.choice(ASPECTS), adj=random.choice(NEGATIVE_ADJ))
        else:
            template = random.choice(NEUTRAL_TEMPLATES)
            feedback = template.format(aspect=random.choice(ASPECTS))

        data.append({'intern_id': intern_id, 'feedback_text': feedback})

    return data


if __name__ == '__main__':
    # quick demo
    samples = generate_feedback(10)
    for s in samples:
        print(s)