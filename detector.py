import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def analyze_text(text: str):
    """
    Analyzes input text using statistical NLP metrics:
    1. Sentence length variance (Burstiness)
    2. Type-Token Ratio (Vocabulary Diversity)
    3. LLM Transition Word Density
    """
    if not text.strip():
        return {"error": "Empty text provided"}

    # Tokenize into sentences and words
    sentences = sent_tokenize(text)
    words = [w.lower() for w in word_tokenize(text) if w.isalnum()]

    if not sentences or not words:
        return {"error": "Invalid text input"}

    
    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    
    
    variance = sum((l - avg_sentence_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)

    
    unique_words = len(set(words))
    total_words = len(words)
    ttr = unique_words / total_words if total_words > 0 else 0

    
    ai_transition_words = ["furthermore", "delve", "testament", "pivotal", "in conclusion", "crucial", "imperative"]
    flagged_words = [w for w in words if w in ai_transition_words]

    return {
        "total_sentences": len(sentences),
        "total_words": total_words,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "sentence_length_variance": round(variance, 2),
        "vocabulary_diversity_ttr": round(ttr, 3),
        "flagged_ai_transitions": flagged_words
    }