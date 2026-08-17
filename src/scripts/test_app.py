import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Streamlit Deployment Test",
    layout="centered"
)

# Header
st.title("Streamlit Connection Test")
st.subheader("Repository Deployment Verification")

# Verification Banner
st.success("Success! Your app is live and connected to GitHub.")

st.markdown("---")