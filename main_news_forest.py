from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

data_path = "../hey-rick-this-looks-like-a-complex-fake"
train_file = "train_B_text.csv"
test_file = "test_B_text.csv"

train_data = pd.read_csv(f"{data_path}/{train_file}")
test_data = pd.read_csv(f"{data_path}/{test_file}")
X = train_data['Title']
y = train_data['Fake/Real']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using CountVectorizer (you can also use TfidfVectorizer)
vectorizer = CountVectorizer(min_df=1, stop_words='english', strip_accents='unicode')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_vectorized, y_train)

# Predict
y_pred = rf.predict(X_test_vectorized)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Predict on test set
rf.fit(vectorizer.transform(X), y)

X_test_final = test_data['Title']
X_test_final_vectorized = vectorizer.transform(X_test_final)
y_test_final_pred = rf.predict(X_test_final_vectorized)

print("y_test_final_pred", y_test_final_pred)

# Save predictions to CSV
df_predictions = pd.DataFrame(y_test_final_pred, columns=['Prediction'])
df_predictions.to_csv("predictions_2.csv", index=False)