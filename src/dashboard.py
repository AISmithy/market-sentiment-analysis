import streamlit as st
from src.data_ingestion import get_stock_data, get_company_info, get_stock_news
from src.widgets.sidebar_widget import render_sidebar
from src.widgets.price_chart_widget import render_price_chart
from src.widgets.company_profile_widget import render_company_profile
from src.widgets.news_sentiment_widget import render_news_sentiment

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
ticker_symbol = render_sidebar()

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
            render_price_chart(hist_data)

        with tab2:
            render_company_profile(company_info)

        with tab3:
            render_news_sentiment(news)
    else:
        st.error(f"Could not retrieve complete data for ticker '{ticker_symbol}'.")
else:
    st.info("Enter a stock ticker in the sidebar to get started.")