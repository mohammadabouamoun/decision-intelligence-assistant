import re

def extract_features_for_ml(text: str) -> dict:
    text = str(text)
    urgent_keywords = ['refund', 'broken', 'cancel', 'down', 'help', 'urgent', 'problem', 'issue', 'not working', 'error', 'complaint', 'charge', 'money', 'lost', 'stolen', 'urgently', 'asap', 'immediately', 'emergency', 'critical', 'frustrated', 'angry', 'disappointed']
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'urgent_keyword_count': sum(1 for kw in urgent_keywords if kw in text.lower()),
        'has_urgent_keyword': int(any(kw in text.lower() for kw in urgent_keywords))
    }