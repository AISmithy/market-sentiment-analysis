import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_news_sentiment(news):
    st.subheader("Latest News & AI Sentiment Analysis")
    if news:
        sentiment_df = pd.DataFrame(news)
        sentiment_counts = sentiment_df['sentiment_label'].value_counts()

        colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

        pie_fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=.3,
            marker_colors=[colors[label] for label in sentiment_counts.index if label in colors]
        )])
        pie_fig.update_layout(title_text='Recent News Sentiment Distribution')
        st.plotly_chart(pie_fig, use_container_width=True)
        st.divider()

        sentiment_emojis = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🔵"}

        for item in news:
            emoji = sentiment_emojis.get(item['sentiment_label'], "⚫")
            st.markdown(f"**{emoji} [{item['title']}]({item['link']})**")
            score_display = f"({item['sentiment_score']:.2f})"
            st.write(f"_{item['published']}_ | **Sentiment:** {item['sentiment_label']} {score_display}")
            st.divider()
    else:
        st.write("No news found or model failed to analyze.")
