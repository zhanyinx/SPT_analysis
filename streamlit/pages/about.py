import streamlit as st


def about():
    st.title(
        "This app allows you to navigate through the single particle tracking data."
    )
    st.markdown(
        """
		### What you can find:
		
		- Display systematic error and rouse time tables

		- Mean square displacement analysis 

		- Distribution of displacement angles

        - Analysis of two color imaging (pairwise distance)

        ### Abbreviation
        
        - EATAmsd: ensamble averaged time averaged mean square displacement
        
        - tamsd: time averaged mean square displacement
		
		"""
    )
