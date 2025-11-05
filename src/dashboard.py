"""\
© 2025 Nishant Kumar. Confidential and Proprietary.
Unauthorized copying, distribution, modification, or use of this software
is strictly prohibited without express written permission.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_ingestion import get_stock_data, get_company_info, get_stock_news

#Author - Nishant

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Analyzer",
    page_icon="💹",
    layout="wide"
)

# --- Page Title ---
st.title("Financial Analyzer")

# --- Sidebar for User Input ---
st.sidebar.header("User Input")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "TSLA").upper()

# --- Main Page Content ---
if ticker_symbol:
    company_info = get_company_info(ticker_symbol)
    hist_data = get_stock_data(ticker_symbol)
    news = get_stock_news(ticker_symbol)

    if company_info and (hist_data is not None and not hist_data.empty):

        st.header(f"{company_info['Company Name']} ({ticker_symbol})")

        # --- Tabbed Interface ---
        tab1, tab2, tab3 = st.tabs(["Price Chart", "Company Profile", "News & Sentiment"])

        with tab1:
            st.subheader("Historical Price Data (Candlestick)")
            fig = go.Figure(data=[go.Candlestick(x=hist_data.index,
                                                 open=hist_data['Open'],
                                                 high=hist_data['High'],
                                                 low=hist_data['Low'],
                                                 close=hist_data['Close'])])
            fig.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Company Profile")
            st.markdown(f"**Sector:** {company_info['Sector']}")
            st.markdown(f"**Market Cap:** {company_info['Market Cap']}")
            st.subheader("Business Summary")
            st.write(company_info['Business Summary'])

        with tab3:
            st.subheader("Latest News & AI Sentiment Analysis")
            if news:
                # --- Sentiment Summary Chart ---
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

                # --- Detailed News List with Sentiment ---
                sentiment_emojis = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🔵"}

                for item in news:
                    emoji = sentiment_emojis.get(item['sentiment_label'], "⚫")
                    st.markdown(f"**{emoji} [{item['title']}]({item['link']})**")
                    score_display = f"({item['sentiment_score']:.2f})"
                    st.write(f"_{item['published']}_ | **Sentiment:** {item['sentiment_label']} {score_display}")
                    st.divider()
            else:
                st.write("No news found or model failed to analyze.")
    else:
        st.error(f"Could not retrieve complete data for ticker '{ticker_symbol}'.")
else:
    st.info("Enter a stock ticker in the sidebar to get started.")