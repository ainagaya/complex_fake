import pandas as pd
import numpy as np
from sklearn import tree, metrics
import matplotlib.pyplot as plt

def load_data(data_path, train_file, test_file):
    train_data = pd.read_csv(f"{data_path}/{train_file}")
    test_data = pd.read_csv(f"{data_path}/{test_file}")
    return train_data, test_data

def preprocess_data(train_data, test_data):
    # Combine redundant columns
    for col in ['Small', 'Mid', 'Large']:
        train_data[col] = train_data[f'{col} size'].combine_first(train_data[col])
        test_data[col] = test_data[f'{col} size'].combine_first(test_data[col])
    
    # Drop redundant columns
    train_data.drop(columns=['Small size', 'Mid size', 'Large size'], inplace=True)
    test_data.drop(columns=['Small size', 'Mid size', 'Large size'], inplace=True)
    
    # Impute missing values
    impute_values = {
        'Genetic Propensity': train_data['Genetic Propensity'].median(),
        'Skin X test': train_data['Skin X test'].mean(),
        'Mid': 0, 'Small': 0, 'Large': 0,
        'Lession': 0, 'Skin color': 0.5
    }
    
    for col, value in impute_values.items():
        train_data[col].fillna(value, inplace=True)
        test_data[col].fillna(value, inplace=True)
    
    # Drop unnecessary columns
    train_data.drop(columns=['Doughnuts consumption'], inplace=True)
    test_data.drop(columns=['Doughnuts consumption'], inplace=True)
    
    return train_data, test_data

def split_data(data, validation_size):
    X = data[:, 1:-1]
    y = np.where(data[:, -1] == 'fake', 0, 1)
    return X[validation_size:], X[:validation_size], y[validation_size:], y[:validation_size]

def train_and_evaluate(X_train, y_train, X_test, y_test, max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    return clf, predictions, accuracy

def plot_tree(clf, feature_names):
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=True)
    plt.show()

def main():
    data_path = "../hey-rick-this-looks-like-a-complex-fake"
    train_file = "train_A_derma.csv"
    test_file = "test_A_derma.csv"
    max_depth = 4
    
    train_data, test_data = load_data(data_path, train_file, test_file)
    train_data, test_data = preprocess_data(train_data, test_data)
    
    size_validation = int(train_data.shape[0] * 0.3)
    data = np.asarray(train_data)
    X_train, X_test, y_train, y_test = split_data(data, size_validation)
    
    #for max_depth in range(1, 10):
    #    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    #    clf.fit(X_train, y_train)
    #    predictions = clf.predict(X_test)
    #    accuracy = metrics.accuracy_score(y_test, predictions)
    #    print(f"Max depth: {max_depth}, Accuracy: {accuracy}")
    
    clf, predictions, accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, max_depth)
    print(f"Accuracy: {accuracy}")
    
    #plot_tree(clf, train_data.columns[1:-1])
    
    # Train on full data and predict on test set
    clf.fit(data[:, 1:-1], np.where(data[:, -1] == 'fake', 0, 1))
    test_data_array = np.asarray(test_data)
    test_predictions = clf.predict(test_data_array[:, 1:])
    test_predictions = np.where(test_predictions == 0, 'fake', 'real')
    
    pd.DataFrame(test_predictions, columns=['Prediction']).to_csv("predictions_1.csv", index=False)
    
if __name__ == "__main__":
    main()
