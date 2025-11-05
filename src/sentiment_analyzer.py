"""\
© 2025 Nishant Kumar. Confidential and Proprietary.
Unauthorized copying, distribution, modification, or use of this software
is strictly prohibited without express written permission.
"""

from transformers import pipeline

# Try to import streamlit cache decorator; fall back to noop when not available.
try:
    import streamlit as st
    cache_resource = st.cache_resource
    _st_present = True
except Exception:
    _st_present = False
    def cache_resource(fn):
        return fn


@cache_resource
def load_sentiment_model():
    """
    Loads the FinBERT sentiment analysis model.
    """
    try:
        # Load the specialized financial sentiment model
        model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        return model
    except Exception as e:
        if _st_present:
            st.error(f"Error loading sentiment model: {e}")
        else:
            print(f"Error loading sentiment model: {e}")
        return None


def analyze_sentiment(text, classifier):
    """
    Analyzes the sentiment of a given text using the loaded model.
    """
    if not classifier or not text:
        return {'label': 'neutral', 'score': 0.0}

    try:
        # The model requires the text to be a list
        results = classifier([text])

        # Remap labels for consistent display
        sentiment_map = {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}
        result = results[0]
        result['label'] = sentiment_map.get(result['label'], 'Neutral')
        return result
    except Exception as e:
        return {'label': 'neutral', 'score': 0.0}