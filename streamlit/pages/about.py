import streamlit as st


def about():
    st.title(
        "This app allows you to navigate through the single particle tracking data."
    )
    st.markdown(
        """
		What you can find:
		
		- Display systematic error and rouse time tables

		- Mean square displacement analysis 

		- Distribution of displacement angles
		
		"""
    )
