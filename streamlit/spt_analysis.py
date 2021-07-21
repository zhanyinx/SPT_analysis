import streamlit as st
from pages.visualize_msd import visualize_msd
from pages.about import about

PAGES = {"About": "about", "Mean square displacement analysis": "msd"}

selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if PAGES[selection] == "about":
    about()

if PAGES[selection] == "msd":
    visualize_msd()
