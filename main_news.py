import pandas as pd
from IPython.display import display

import numpy as np

data_path = "../hey-rick-this-looks-like-a-complex-fake"

train_B_text = data_path + "/" + "train_B_text.csv"
test_B_text = data_path + "/" + "test_B_text.csv"

print("train_B_text: ")
print(train_B_text)

dfa_train_B = pd.read_csv(train_B_text)
dfa_test_B = pd.read_csv(test_B_text)

size_train_B = dfa_train_B.shape

# Remove the Id
dfa_train_B = dfa_train_B.drop(columns=['Id'])
dfa_test_B = dfa_test_B.drop(columns=['Id'])

display(dfa_train_B)

# Split into train and validation
size_validation = int(size_train_B[0] * 0.3)

data = np.asarray(dfa_train_B)
data_test = np.asarray(dfa_test_B)

X_train = data[:size_train_B[0] - size_validation, :-1]
# X_train is a list of lists, convert it to a list of strings
X_train = [str(x[0]) for x in X_train]

Y_train = data[:size_train_B[0] - size_validation, -1]

X_validation = data[size_train_B[0] - size_validation:, :-1]
X_validation = [str(x[0]) for x in X_validation]
Y_validation = data[size_train_B[0] - size_validation:, -1]

from sklearn.feature_extraction.text import CountVectorizer

# We use the count number of instances considering that a word has a minimum support of two documents
vectorizer = CountVectorizer(min_df=2, 
# stop words such as 'and', 'the', 'of' are removed                             
 stop_words='english', 
 strip_accents='unicode')

#example of the tokenization
test_string = X_train[0]
print ("Example: " + test_string)
print ("Preprocessed: " + vectorizer.build_preprocessor()(test_string))
print ("Tokenized:" + str(vectorizer.build_tokenizer()(test_string)))
print ("Analyzed data string:" + str(vectorizer.build_analyzer()(test_string)))
print("-----------------------------------")

#Process and convert data
X_train = vectorizer.fit_transform(X_train)
X_validation = vectorizer.transform(X_validation)

print ("Number of tokens: " + str(len(vectorizer.get_feature_names_out())) +"\n")
print ("Extract of tokens:")
print (vectorizer.get_feature_names_out()[1000:1100])

from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB()
nb.fit(X_train,Y_train)

prediction = nb.predict(X_validation)

from sklearn.metrics import accuracy_score
print ("Accuracy: " + str(accuracy_score(Y_validation, prediction)))

# Now let's train it with the whole data, and test it with the test data
X_train = data[:, :-1]
X_train = [str(x[0]) for x in X_train]
Y_train = data[:, -1]
X_train = vectorizer.fit_transform(X_train)

X_test = data_test
X_test= [str(x[0]) for x in X_test]
X_test = vectorizer.transform(X_test)

nb = BernoulliNB()
nb.fit(X_train,Y_train)

prediction = nb.predict(X_test)
print("predictions: ", prediction)

# Save the predictions
df_predictions = pd.DataFrame(prediction)
df_predictions.columns = ['Prediction']
df_predictions.to_csv("predictions_2.csv", index=False, header=True)