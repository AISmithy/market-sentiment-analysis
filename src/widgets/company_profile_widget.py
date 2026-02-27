import streamlit as st


def render_company_profile(company_info):
    st.subheader("Company Profile")
    st.markdown(f"**Sector:** {company_info['Sector']}")
    st.markdown(f"**Market Cap:** {company_info['Market Cap']}")
    st.subheader("Business Summary")
    st.write(company_info['Business Summary'])
