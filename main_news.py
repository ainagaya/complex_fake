import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def load_data(data_path, train_file, test_file):
    train_data = pd.read_csv(f"{data_path}/{train_file}")
    test_data = pd.read_csv(f"{data_path}/{test_file}")
    return train_data, test_data

def preprocess_data(train_data, test_data):
    train_data = train_data.drop(columns=['Id'])
    test_data = test_data.drop(columns=['Id'])
    return train_data, test_data

def split_data(data, validation_size):
    data_array = np.asarray(data)
    X = data_array[:, :-1]
    Y = data_array[:, -1]
    X_train = X[:-validation_size]
    Y_train = Y[:-validation_size]
    X_validation = X[-validation_size:]
    Y_validation = Y[-validation_size:]
    return X_train, Y_train, X_validation, Y_validation

# def preprocess_with_stemming(text):
#     words = word_tokenize(text)  # Tokenize the text
#     stemmed_words = [stemmer.stem(word) for word in words]  # Apply stemming
#     return ' '.join(stemmed_words)  # Join words back into a single string

def vectorize_data(X_train, X_validation, min_df=2, stop_words='english', strip_accents='unicode'):
    # Initialize stemmer
    #global stemmer
    #stemmer = PorterStemmer()
    #X_train_preprocessed = [preprocess_with_stemming(str(x[0])) for x in X_train]
    #X_validation_preprocessed = [preprocess_with_stemming(str(x[0])) for x in X_validation]
    #print("X_train", X_train[10:20])
    #print("X_train_preprocessed", X_train_preprocessed[10:20])
    vectorizer = CountVectorizer(min_df=min_df, stop_words=stop_words, strip_accents=strip_accents)
    #X_train = vectorizer.fit_transform([str(x[0]) for x in X_train_preprocessed])
    #X_validation = vectorizer.transform([str(x[0]) for x in X_validation_preprocessed])
    X_train = vectorizer.fit_transform([str(x[0]) for x in X_train])
    X_validation = vectorizer.transform([str(x[0]) for x in X_validation])
    return X_train, X_validation, vectorizer

def train_and_evaluate(X_train, Y_train, X_validation, Y_validation):
    nb = BernoulliNB()
    nb.fit(X_train, Y_train)
    prediction = nb.predict(X_validation)
    accuracy = accuracy_score(Y_validation, prediction)
    return nb, accuracy

def main():
    data_path = "../hey-rick-this-looks-like-a-complex-fake"
    train_file = "train_B_text.csv"
    test_file = "test_B_text.csv"

    # Download necessary NLTK data
    nltk.download('punkt')
    nltk.download('punkt_tab')

    

    train_data, test_data = load_data(data_path, train_file, test_file)
    display(train_data)

    train_data, test_data = preprocess_data(train_data, test_data)
    size_validation = int(train_data.shape[0] * 0.3)

    X_train, Y_train, X_validation, Y_validation = split_data(train_data, size_validation)
    X_train, X_validation, vectorizer = vectorize_data(X_train, X_validation)

    nb, accuracy = train_and_evaluate(X_train, Y_train, X_validation, Y_validation)
    print(f"Accuracy: {accuracy}")

    # Train with the whole data and test with the test data
    X_train = np.asarray(train_data)[:, :-1]
    Y_train = np.asarray(train_data)[:, -1]
    X_train = vectorizer.fit_transform([str(x[0]) for x in X_train])

    X_test = np.asarray(test_data)
    X_test = vectorizer.transform([str(x[0]) for x in X_test])

    nb.fit(X_train, Y_train)
    prediction = nb.predict(X_test)
    print("predictions: ", prediction)

    # Save the predictions
    df_predictions = pd.DataFrame(prediction, columns=['Prediction'])
    df_predictions.to_csv("predictions_2.csv", index=False, header=True)

if __name__ == "__main__":
    main()
