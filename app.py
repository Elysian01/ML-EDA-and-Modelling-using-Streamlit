import streamlit as st

from multiapp import MultiApp
from apps import eda
from apps import models

st.set_page_config("Data Explorer & Modeller")

app = MultiApp()

app.add_app("Data Explorer", eda.app)
app.add_app("Machine Learning Modelling", models.app)
app.run()
