import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from .utils import select_or_upload_dataset
from apps.algo.logistic import LogisticRegression, accuracy
from apps.algo.knn import K_Nearest_Neighbors_Classifier
from apps.algo.naive import Naive
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

naive_model = Naive()
lr = LogisticRegression(lr=0.001, epochs=100)


def modelling(df):

    show_custom_predictions = True  # for show custom predictions checkbox

    # Show Dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Numbers of rows to view", 5)
        st.dataframe(df.head(number))

    # Select Algorithm
    algos = [
        "Logistic Regression For Binary Classification",
        "Naive Bayes",
        "K Nearest Neighbors Classifier",
    ]
    selected_algo = st.selectbox("Select Algorithm", algos)

    if selected_algo:
        st.info("You Selected {} Algorithm".format(selected_algo))

    # Select the Taget Feature
    all_columns = df.columns
    target_column = st.selectbox("Select Target Column", all_columns)
    target_data = df[target_column]

    if target_column:
        if st.checkbox("Show Value Count of Target Feature"):
            st.write(df[target_column].value_counts())

    # Label Encoding
    label_encoder = preprocessing.LabelEncoder()
    encoded_target_data = label_encoder.fit_transform(target_data)
    encoded_target_data_knn = pd.DataFrame(encoded_target_data, columns=[target_column])
    st.info("You Selected {} Column".format(target_column))
    remain_column = df.drop([target_column], axis=1)

    if len(df.eval(target_column).unique()) > 25:
        st.error(
            "Please select classification dataset only, or the target column must be categorical"
        )
    
    else:

        # Select the Remaining Feature and scaling them
        X = df.drop([target_column], axis=1)
        y = encoded_target_data
        y_knn = encoded_target_data_knn.values
        X_knn = remain_column.values
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

            if selected_algo == "Logistic Regression For Binary Classification":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=1234
                )

                if len(set(y)) == 2:
                    lr.fit(X_train, y_train)
                    show_custom_predictions = True

                    # making predictions on the testing set
                    y_pred = lr.predict(X_test)
                    print("Accuracy: ", accuracy(y_test, y_pred))

                    # comparing actual response values (y_test) with predicted response values (y_pred)
                    st.write(
                        "Model accuracy of Logistic Regression Model:",
                        metrics.accuracy_score(y_test, y_pred) * 100,
                    )
                    output(y_test, y_pred)
                else:
                    show_custom_predictions = False
                    st.error(
                        "The Target feature has more than 2 classes, please select another dataset or a different classification algorithm."
                    )

            if selected_algo == "K Nearest Neighbors Classifier":
                X_train, X_test, y_train, y_test = train_test_split(
                    X_knn, y_knn, test_size=test_size, random_state=1234
                )

                # Model training
                if k_size != 0:
                    knn_model = K_Nearest_Neighbors_Classifier(K=k_size)
                    knn_model.fit(X_train, y_train)

                    # Prediction on test set
                    Y_pred = knn_model.predict(X_test)

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

                else:
                    st.error("Cannot predict of K=0")

            if selected_algo == "Naive Bayes":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=1234
                )

                naive_model.fit(X_train, y_train)
                Y_pred = naive_model.predict(X_test)

                # comparing actual response values (y_test) with predicted response values (y_pred)
                st.write(
                    "Model accuracy of Naive Bayes Model:",
                    metrics.accuracy_score(y_test, Y_pred) * 100,
                )
                output(y_test, Y_pred)

        # Make Custom Prediction
        if show_custom_predictions:
            if st.checkbox("Make custom prediction"):
                count = 0
                storevalues = []
                for i in range(len(df.columns) - 1):
                    takeinput = st.number_input(
                        df.columns[count],
                        help=f"Example: {df.iloc[2][count]}",
                        key=count,
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

                    if selected_algo == "Naive Bayes":
                        X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=1234
                )
                        gnb = GaussianNB()
                        gnb.fit(X_train, y_train)

                # making predictions on the testing set
                        Y_pred = gnb.predict(storevalues)
                        # making predictions on the testing set
                        
                        st.write(
                            "Predicted value for the given custom data :",
                            label_encoder.inverse_transform(np.array(Y_pred)),
                        )

                    if selected_algo == "Logistic Regression For Binary Classification":
                        y_pred = lr.predict(storevalues)
                        st.write(
                            "Predicted value for the given custom data :",
                            label_encoder.inverse_transform(np.array(y_pred)),
                        )

                    if selected_algo == "K Nearest Neighbors Classifier":
                        X_train, X_test, y_train, y_test = train_test_split(
                    X_knn, y_knn, test_size=test_size, random_state=1234
                )
                        
                        knn_models = KNeighborsClassifier(n_neighbors=k_size)

                        knn_models.fit(X_train, y_train)

                        y_pred = knn_models.predict(storevalues)

                        st.write(
                            "Predicted value for the given custom data :",
                            label_encoder.inverse_transform(np.array(y_pred)),
                        )


def output(y_test, y_pred):
    confusion_matrix_out = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix_out)

    st.write("Classification Report:")
    precision_score_out = precision_score(y_test, y_pred)
    recall_score_out = recall_score(y_test, y_pred)
    f1_score_out = f1_score(y_test, y_pred)

    data = [
        ["Precison Score", precision_score_out],
        ["Recall Score", recall_score_out],
        ["f1 Score", f1_score_out],
    ]
    df = pd.DataFrame(data, columns=["Parameter", "Value"])
    st.table(df)


def app():
    st.title("Machine Learning Modelling")
    st.subheader("Choose your dataset ")
    select_or_upload_dataset(modelling)


if __name__ == "__main__":
    app()
