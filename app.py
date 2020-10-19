from os import kill
from notes.main import main
import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def main():
    st.title("ML Dataset Explorer")
    st.subheader("Simple Data Science Explorer")

    def file_selector(folder_path = "."):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Select a data file",filenames)

        return os.path.join(folder_path,selected_filename)

    filename = file_selector("./datasets")
    formatted_filename = filename.split("/")[2].split(".")[0].title()
    st.info("You Selected {} Dataset".format(formatted_filename))

    # Read Data
    df = pd.read_csv(filename)

    # Show Dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Numbers of rows to view",1)
        st.dataframe(df.head(number))

    # Show Columns
    if st.checkbox("Columns Names"):
        st.write(df.columns)
    
    # Show Shape
    if st.checkbox("Shape of Dataset"):
        st.write(df.shape)
        data_dim = st.radio("Show Dimension by ",("Rows","Columns"))
        if data_dim == "Columns":
            st.text("Numbers of Columns")
            st.write(df.shape[1])
        elif data_dim == "Rows":
            st.text("Numbers of Rows")
            st.write(df.shape[0])
        else:
            st.write(df.shape)
        
    
    # Select Columns
    if st.checkbox("Select Column to show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select Columns",all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    # Show Value Count
    if st.checkbox("Show Value Counts"):
        st.text("Value Counts by Target/Class")
        st.write(df.iloc[:,-1].value_counts())

    # Show Datatypes
    if st.checkbox("Show Data types"):
        st.text("Data Types")
        st.write(df.dtypes)

    # Show Summary
    if st.checkbox("Show Summary"):
        st.text("Summary")
        st.write(df.describe().T)


    # Plot and visualization
    st.subheader("Data Visualization")

    ## Correlation Seaborn Plot
    if st.checkbox("Show Correlation Plot"):
        all_columns_names = df.columns.tolist()
        st.success("Generating Correlation Plot ...")
        st.write(sns.heatmap(df.corr(),annot=True))
        st.pyplot()

    ## Count Plot
    if st.checkbox("Show Value Count Plots"):
        st.text("Value count by target")
        all_columns_names = df.columns.tolist()
        primary_column = st.selectbox("Primary Column to Groupby",all_columns_names)
        selected_columns_names = st.multiselect("Select Columns",all_columns_names)
        st.success("Generating Plot ...")
        if selected_columns_names:
            vc_plot = df.groupby(primary_column)[selected_columns_names].count()
        else:
            vc_plot = df.iloc[:,-1].value_counts()
        st.write(vc_plot.plot(kind="bar"))
        st.pyplot()


    ## Pie Chart
    if st.checkbox("Show Pie Plot"):
        all_columns_names = df.columns.tolist()
        st.success("Generating Pie Chart ...")
        st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
        st.pyplot()

    ## Customizable Plot
    st.subheader("Customizable Plot")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select type of Plot",["area","bar","line","hist","box","kde"])
    selected_columns_names = st.multiselect("Select Columns to plot",all_columns_names)

    if st.button("Generate Plot"):
        st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

        custom_data = df[selected_columns_names]
        if type_of_plot == "area":
            st.area_chart(custom_data)

        elif type_of_plot == "bar":
            st.bar_chart(custom_data)

        elif type_of_plot == "line":
            st.line_chart(custom_data)

        elif type_of_plot:
            custom_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

    # st.balloons()
if __name__ == "__main__":
    main()
