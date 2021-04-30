import matplotlib
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from .utils import select_or_upload_dataset

matplotlib.use("Agg")
fig, ax = plt.subplots()
matplotlib.rcParams.update({'font.size': 8})


def categorical_column(df, max_unique_values=15):
    categorical_column_list = []
    for column in df.columns:
        if df[column].nunique() < max_unique_values:
            categorical_column_list.append(column)
    return categorical_column_list


def eda(df):

    # Show Dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Numbers of rows to view", 5)
        st.dataframe(df.head(number))

    # Show Columns
    if st.checkbox("Columns Names"):
        st.write(df.columns)

    # Show Shape
    if st.checkbox("Shape of Dataset"):
        st.write(df.shape)
        data_dim = st.radio("Show Dimension by ", ("Rows", "Columns"))
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
        selected_columns = st.multiselect("Select Columns", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    # Show Value Count
    if st.checkbox("Show Value Counts"):
        all_columns = df.columns.tolist()
        selected_columns = st.selectbox("Select Column", all_columns)
        st.write(df[selected_columns].value_counts())

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
    all_columns_names = df.columns.tolist()

    # Correlation Seaborn Plot
    if st.checkbox("Show Correlation Plot"):
        st.success("Generating Correlation Plot ...")
        if st.checkbox("Annot the Plot"):
            st.write(sns.heatmap(df.corr(), annot=True))
        else:
            st.write(sns.heatmap(df.corr()))
        st.pyplot()

    # Count Plot
    if st.checkbox("Show Value Count Plots"):
        x = st.selectbox("Select Categorical Column", all_columns_names)
        st.success("Generating Plot ...")
        if x:
            if st.checkbox("Select Second Categorical column"):
                hue_all_column_name = df[df.columns.difference([x])].columns
                hue = st.selectbox(
                    "Select Column for Count Plot", hue_all_column_name)
                st.write(sns.countplot(x=x, hue=hue, data=df, palette="Set2"))
            else:
                st.write(sns.countplot(x=x, data=df, palette="Set2"))
            st.pyplot()

    # Pie Chart
    if st.checkbox("Show Pie Plot"):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        all_columns = categorical_column(df)
        selected_columns = st.selectbox("Select Column", all_columns)
        if selected_columns:
            st.success("Generating Pie Chart ...")
            st.write(df[selected_columns].value_counts().plot.pie(
                autopct="%1.1f%%"))
            st.pyplot()

    # Customizable Plot
    st.subheader("Customizable Plot")

    type_of_plot = st.selectbox("Select type of Plot", [
                                "area", "bar", "line", "hist", "box", "kde"])
    selected_columns_names = st.multiselect(
        "Select Columns to plot", all_columns_names)

    if st.button("Generate Plot"):
        st.success("Generating Customizable Plot of {} for {}".format(
            type_of_plot, selected_columns_names))

        custom_data = df[selected_columns_names]
        if type_of_plot == "area":
            st.area_chart(custom_data)

        elif type_of_plot == "bar":
            st.bar_chart(custom_data)

        elif type_of_plot == "line":
            st.line_chart(custom_data)

        elif type_of_plot:
            custom_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(custom_plot)
            st.pyplot()

    # st.balloons()


def app():

    st.title("ML Dataset Explorer")
    st.subheader("Simple Data Science Explorer")

    algo_visualization_template = """
        <h3 style="text-align:center; color: #FF1493;"><a href = "https://elysian01.github.io/Algorithms-Visualization/" target = "_blank">Click here </a> To learn about various algorithms, data structure , encryption and much more </h3><br>
    """

    st.markdown(algo_visualization_template, unsafe_allow_html=True)

    select_or_upload_dataset(eda)


if __name__ == "__main__":
    app()
