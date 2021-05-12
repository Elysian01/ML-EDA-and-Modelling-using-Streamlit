import streamlit as st
import pandas as pd
import os


def select_or_upload_dataset(callback_function):
    """User can upload his own dataset or choose from the dataset provided by the system.

    Args:
        callback_function (function): function should be called after dataset is uploaded.
    """
    dataframe_selection = st.radio("Please select or upload your dataset", (
        "Choose from popular dataset", "Upload my own dataset"))
    df = pd.DataFrame()
    # df = None

    if dataframe_selection == "Choose from popular dataset":
        def file_selector(folder_path="."):
            filenames = os.listdir(folder_path)
            selected_filename = st.selectbox("Select a data file", filenames)

            return os.path.join(folder_path, selected_filename)

        filename = file_selector("./datasets")
        head, formatted_filename = os.path.split(filename)
        # formatted_filename = filename.split("/")[2].split(".")[0].title()
        st.info("You Selected {} Dataset".format(formatted_filename))

        # Read Data
        df = pd.read_csv(filename)
        df.dropna(inplace=True)
        callback_function(df)  # Execute callback function

    elif dataframe_selection == "Upload my own dataset":
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
        if data is not None:
            data.seek(0)
            st.success("Uploaded Dataset Successfully")
            df = pd.read_csv(data)
            df.dropna(inplace=True)
            callback_function(df)  # Execute callback function
