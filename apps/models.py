import streamlit as st

from .utils import select_or_upload_dataset
from apps.algo.logistic import Logistic
from apps.algo.knn import Knn
from apps.algo.naive import Naive


def modelling(df):

    algos = ["Logistic Regression", "Naive Bayes", "K-NN"]
    selected_algo = st.selectbox("Select Algorithm", algos)

    if selected_algo:
        st.info("You Selected {} Algorithm".format(selected_algo))

    if st.button("Start Training", help="Training will start for the selected algorithm on dataset"):
        st.write("Training started...")


def app():
    st.title("Machine Learning Modelling")
    st.subheader("Choose your dataset ")

    select_or_upload_dataset(modelling)


if __name__ == "__main__":
    app()
