import streamlit as st
from pages.visualize_msd import visualize_msd
from pages.about import about
from pages.tables import tables
from pages.dictionary import PAGES


selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if PAGES[selection] == "about":
    about()

if PAGES[selection] == "msd":
    visualize_msd()

if PAGES[selection] == "tables":
    tables()
