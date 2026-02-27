import streamlit as st


def render_sidebar() -> str:
    st.sidebar.header("User Input")
    return st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "TSLA").upper()
