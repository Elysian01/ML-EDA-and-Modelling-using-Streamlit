import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from .utils import select_or_upload_dataset
from apps.algo.logistic import LogisticRegression, accuracy
from apps.algo.knn import K_Nearest_Neighbors_Classifier

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
lr = LogisticRegression(lr=0.001, epochs=100)


def modelling(df):

    # Show Dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Numbers of rows to view", 5)
        st.dataframe(df.head(number))

    # Select Algorithm
    algos = ["Logistic Regression", "Naive Bayes", "K Nearest Neighbors Classifier"]
    selected_algo = st.selectbox("Select Algorithm", algos)

    if selected_algo:
        st.info("You Selected {} Algorithm".format(selected_algo))

    # Select the Taget Feature
    all_columns = df.columns
    target_column = st.selectbox("Select Target Column", all_columns)
    target_data = df[target_column]

    # Label Encoding
    label_encoder = preprocessing.LabelEncoder()
    encoded_target_data = label_encoder.fit_transform(target_data)
    # encoded_target_data = pd.DataFrame(encoded_target_data, columns=[target_column])
    st.info("You Selected {} Column".format(target_column))

    if len(df.eval(target_column).unique()) > 25:
        st.error(
            "Please select classification dataset only, or the target column must be categorical"
        )

    else:

        # Select the Remaining Feature and scaling them
        X = df.drop([target_column], axis=1)
        y = encoded_target_data
        X_names = X.columns
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X, columns=X_names)

        # Train and Test Split
        test_size = st.slider(
            "Select the size of Test Dataset (train test split)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            help="Set the ratio for splitting the dataset into Train and Test Dataset",
        )

        # Set the size of K (only for K-NN)
        if selected_algo == "K Nearest Neighbors Classifier":
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

            # Splitting dataset into train and test set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=1234
            )

            if selected_algo == "Logistic Regression":
                lr.fit(X_train, y_train)

                # making predictions on the testing set
                y_pred = lr.predict(X_test)
                print("Accuracy: ", accuracy(y_test, y_pred))

                # comparing actual response values (y_test) with predicted response values (y_pred)
                st.write(
                    "Model accuracy of Logistic Regression Model:",
                    metrics.accuracy_score(y_test, y_pred) * 100,
                )
                output(y_test, y_pred)

            if selected_algo == "K Nearest Neighbors Classifier":

                # Model training
                if k_size != 0:

                    model = K_Nearest_Neighbors_Classifier(K=k_size)
                    model.fit(X_train, y_train)

                    # Prediction on test set
                    Y_pred = model.predict(X_test)

                    # measure performance
                    correctly_classified = 0

                    # counter
                    count = 0

                    for count in range(np.size(Y_pred)):
                        if y_test[count] == Y_pred[count]:
                            correctly_classified = correctly_classified + 1
                        count = count + 1

                    st.write("Accuracy:")
                    st.info((correctly_classified / count) * 100)
                    output(y_test, Y_pred)

            if selected_algo == "Naive Bayes":
                gnb.fit(X_train, y_train)

                # making predictions on the testing set
                Y_pred = gnb.predict(X_test)

                # comparing actual response values (y_test) with predicted response values (y_pred)
                st.write(
                    "Model accuracy of Naive Bayes Model:",
                    metrics.accuracy_score(y_test, Y_pred) * 100,
                )
                output(y_test, Y_pred)

        # Make Custom Prediction
        if st.checkbox("Make custom prediction"):
            count = 0
            storevalues = []
            for i in range(len(df.columns) - 1):
                takeinput = st.number_input(
                    df.columns[count], help=f"Example: {df.iloc[1][count]}", key=count
                )
                storevalues.append(takeinput)
                count += 1

            # Predict the value
            if st.button(
                "Start Prediction",
                help="Predicting the outcome of the values",
            ):
                storevalues = np.expand_dims(storevalues, axis=0)
                storevalues = scaler.transform(storevalues)
                print(storevalues)

                if selected_algo == "Naive Bayes":
                    # making predictions on the testing set
                    Y_pred = gnb.predict(storevalues)
                    st.write(
                        "Predicted value for the given custom data :",
                        label_encoder.inverse_transform(np.array(Y_pred)),
                    )

                # if selected_algo == "K Nearest Neighbors Classifier":

                #     Y = encoded_target_data.values
                #     X = remain_column.values

                #     # Splitting dataset into train and test set
                #     X_train, X_test, y_train, y_test = train_test_split(
                #         X, Y, test_size=test_size, random_state=0
                #     )

                #     # Model training
                #     if k_size != 0:

                #         from sklearn.neighbors import KNeighborsClassifier

                #         classifier = KNeighborsClassifier(
                #             n_neighbors=k_size, metric="minkowski", p=2
                #         )
                #         classifier.fit(X_train, y_train)

                #         # making predictions on the testing set
                #         Y_pred = classifier.predict(storevalues)

                #         st.write(
                #             "Predicted value for the given custom data :",
                #             Y_pred[0],
                #         )


def output(y_test, y_pred):
    confusion_matrix_out = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix_out)

    classification_report_out = classification_report(y_test, y_pred)
    st.write("Classification Report:")
    st.write(classification_report_out)


def app():
    st.title("Machine Learning Modelling")
    st.subheader("Choose your dataset ")
    select_or_upload_dataset(modelling)


if __name__ == "__main__":
    app()
