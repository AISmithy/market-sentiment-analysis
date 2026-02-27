import streamlit as st
import plotly.graph_objects as go


def render_price_chart(hist_data):
    st.subheader("Historical Price Data (Candlestick)")
    fig = go.Figure(data=[go.Candlestick(x=hist_data.index,
                                         open=hist_data['Open'],
                                         high=hist_data['High'],
                                         low=hist_data['Low'],
                                         close=hist_data['Close'])])
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
