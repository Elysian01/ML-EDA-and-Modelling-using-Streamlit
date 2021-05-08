import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

from .utils import select_or_upload_dataset
from apps.algo.logistic import LogisticRegression
from apps.algo.knn import K_Nearest_Neighbors_Classifier


def modelling(df):

    # Show Dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Numbers of rows to view", 5)
        st.dataframe(df.head(number))

    # Select Algorithm
    algos = ["Logistic Regression", "Naive Bayes", "K_Nearest_Neighbors_Classifier"]
    selected_algo = st.selectbox("Select Algorithm", algos)
    print(selected_algo)

    if selected_algo:
        st.info("You Selected {} Algorithm".format(selected_algo))

    # Select the Taget Feature
    column_name = df.columns
    target_column = st.selectbox("Select Target Column", column_name)
    target_data = df[target_column]

    # Label Encoding
    label_encoder = preprocessing.LabelEncoder()
    encoded_target_data = label_encoder.fit_transform(target_data)
    encoded_target_data = pd.DataFrame(encoded_target_data, columns=[target_column])
    st.info("You Selected {} Column".format(target_column))

    # Select the Remaining Feature
    remain_column = df.drop([target_column], axis=1)

    # Train and Test Split
    test_size = st.slider(
        "Select the size of Test Dataset (train test split)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        help="Set the ratio for splitting the dataset into Train and Test Dataset",
    )

    # Set the size of K (only for K-NN)
    if selected_algo == "K_Nearest_Neighbors_Classifier":
        k_size = st.number_input(
            "Select the size of K",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            help="Set the ratio for splitting the Train and Test Dataset",
        )

    if st.button(
        "Start Training",
        help="Training will start for the selected algorithm on dataset",
    ):
        if selected_algo == "K_Nearest_Neighbors_Classifier":

            Y = encoded_target_data.values
            X = remain_column.values

            # Splitting dataset into train and test set
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=test_size, random_state=0
            )

            # Model training
            if k_size != 0:

                model = K_Nearest_Neighbors_Classifier(K=k_size)
                model.fit(X_train, Y_train)

                # Prediction on test set
                Y_pred = model.predict(X_test)

                # measure performance
                correctly_classified = 0

                # counter
                count = 0

                for count in range(np.size(Y_pred)):
                    if Y_test[count] == Y_pred[count]:
                        correctly_classified = correctly_classified + 1
                    count = count + 1

                st.write("Accuracy:")
                st.info((correctly_classified / count) * 100)
                output(Y_test, Y_pred)

                # Custom Dataset Prediction
                # x = np.array([[5.8000 , 2.8000,5.1000,2.4000]])
                # dataClass = model.predict(x)
                # st.write(dataClass)

        if selected_algo == "Naive Bayes":
            Y = encoded_target_data.values
            X = remain_column.values

            # Splitting dataset into train and test set
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=test_size, random_state=0
            )

            # training the model on training set
            from sklearn.naive_bayes import GaussianNB

            gnb = GaussianNB()
            gnb.fit(X_train, Y_train)

            # making predictions on the testing set
            Y_pred = gnb.predict(X_test)

            # comparing actual response values (y_test) with predicted response values (y_pred)
            from sklearn import metrics

            st.write(
                "Model accuracy:",
                metrics.accuracy_score(Y_test, Y_pred) * 100,
            )
            output(Y_test, Y_pred)


def output(Y_test, Y_pred):
    confusion_matrix_out = confusion_matrix(Y_test, Y_pred)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix_out)

    classification_report_out = classification_report(Y_test, Y_pred)
    st.write("Classification Report:")
    st.write(classification_report_out)


def app():
    st.title("Machine Learning Modelling")
    st.subheader("Choose your dataset ")
    select_or_upload_dataset(modelling)


if __name__ == "__main__":
    app()
