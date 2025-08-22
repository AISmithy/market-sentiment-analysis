from transformers import pipeline
import streamlit as st


@st.cache_resource
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
        st.error(f"Error loading sentiment model: {e}")
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
        # In a production environment, you would log this error
        return {'label': 'neutral', 'score': 0.0}