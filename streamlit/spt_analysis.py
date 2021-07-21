import streamlit as st
from pages.visualize_msd import visualize_msd

PAGES = {"msd_analysis": "visualize_msd"}

selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if selection == "msd_analysis":
    visualize_msd()
