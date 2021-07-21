import streamlit as st
from .dictionary import LIST_SAMPLES


def tables():
    st.title("Mean square displacement analysis")

    # Samples
    list_samples = LIST_SAMPLES
