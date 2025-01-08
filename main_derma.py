
import pandas as pd
from IPython.display import display

import numpy as np
from sklearn.impute import SimpleImputer

from sklearn import tree
from sklearn import metrics

import matplotlib.pyplot as plt

data_path = "../hey-rick-this-looks-like-a-complex-fake"

train_A_derma = data_path + "/" + "train_A_derma.csv"
test_A_derma = data_path + "/" + "test_A_derma.csv"

print("train_A_derma: ")
print(train_A_derma)

dfa_train_A = pd.read_csv(train_A_derma)
dfa_test_A = pd.read_csv(test_A_derma)

size_train_A = dfa_train_A.shape

# Remove the rows where Lession is NaN
#dfa_train_A = dfa_train_A.dropna(subset=['Lession'])

size_validation = int(size_train_A[0] * 0.3)

print("size_train_A: ", size_train_A, " size_validation: ", size_validation)

print("dfa_train_A: ")
display(dfa_train_A)

# There is some redundant columns. 'Small size' and 'small' should represent the same, as well as 'mid' and 'mid size', and 'large' and 'large size'
# We will join the columns 'small' and 'small size', that have the same meaning, keeping only one of them, the one that is not NaN
# We will do the same for 'mid' and 'mid size', and 'large' and 'large size'
dfa_train_A['Small'] = dfa_train_A['Small size'].combine_first(dfa_train_A['Small'])
dfa_train_A['Mid'] = dfa_train_A['Mid size'].combine_first(dfa_train_A['Mid'])
dfa_train_A['Large'] = dfa_train_A['Large size'].combine_first(dfa_train_A['Large'])

# Do the same for the test data
dfa_test_A['Small'] = dfa_test_A['Small size'].combine_first(dfa_test_A['Small'])
dfa_test_A['Mid'] = dfa_test_A['Mid size'].combine_first(dfa_test_A['Mid'])
dfa_test_A['Large'] = dfa_test_A['Large size'].combine_first(dfa_test_A['Large'])

# Now we can remove the columns 'Small size', 'Mid size' and 'Large size'
dfa_train_A = dfa_train_A.drop(columns=['Small size', 'Mid size', 'Large size'])
dfa_test_A = dfa_test_A.drop(columns=['Small size', 'Mid size', 'Large size'])

print("dfa_train_A after processing: ")
display(dfa_train_A)

# Plot the distribution of the column 'skin X test' feature, to see how we can impute the missing values
# plt.hist(dfa_train_A['Skin X test'].dropna(), bins=10, edgecolor='k')
# plt.xlabel('Skin X test')
# plt.ylabel('Frequency')
# plt.title('Distribution of skin X test')
# plt.show()

# # Plot the distribution of the column 'Genetic Propensity' feature, to see how we can impute the missing values
# plt.hist(dfa_train_A['Genetic Propensity'].dropna(), bins=30, edgecolor='k')
# plt.xlabel('Genetic Propensity')
# plt.ylabel('Frequency')
# plt.title('Distribution of Genetic Propensity')
# plt.show()

# # Plot the distrubution of the column "lession" feature, to see how we can impute the missing values
# plt.hist(dfa_train_A['Lession'].dropna(), bins=30, edgecolor='k')
# plt.xlabel('Lession')
# plt.ylabel('Frequency')
# plt.title('Distribution of Lession')
# plt.show()

# # Plot the distrubution of the column "Skin color" feature, to see how we can impute the missing values
# plt.hist(dfa_train_A['Skin color'].dropna(), bins=30, edgecolor='k')
# plt.xlabel('Skuin color')
# plt.ylabel('Frequency')
# plt.title('Distribution of Skin color')
# plt.show()

# Give a dataset of the number of NaNs in each column
print("Number of NaNs in each column: ")
print(dfa_train_A.isnull().sum())

# Curate the data

# Genetic propensity feature: we will use the median to impute the missing values
dfa_train_A['Genetic Propensity'].fillna(dfa_train_A['Genetic Propensity'].median(), inplace=True)
dfa_test_A['Genetic Propensity'].fillna(dfa_train_A['Genetic Propensity'].median(), inplace=True)

# Skin X test feature: we will use the mean to impute the missing values
dfa_train_A['Skin X test'].fillna(dfa_train_A['Skin X test'].mean(), inplace=True)
dfa_test_A['Skin X test'].fillna(dfa_train_A['Skin X test'].mean(), inplace=True)

# Mid, small and large features: we will replace the missing values with 0
dfa_train_A['Mid'].fillna(0, inplace=True)
dfa_train_A['Small'].fillna(0, inplace=True)
dfa_train_A['Large'].fillna(0, inplace=True)
dfa_test_A['Mid'].fillna(0, inplace=True)
dfa_test_A['Small'].fillna(0, inplace=True)
dfa_test_A['Large'].fillna(0, inplace=True)

# Lession feature: we will replace the missing values with 0
dfa_train_A['Lession'].fillna(0, inplace=True)
dfa_test_A['Lession'].fillna(0, inplace=True)

# Skin color feature: we will replace the missing values with 0.5
dfa_train_A['Skin color'].fillna(0.5, inplace=True)
dfa_test_A['Skin color'].fillna(0.5, inplace=True)

# Remove the column "Doughnuts consumption"
dfa_train_A = dfa_train_A.drop(columns=['Doughnuts consumption'])
dfa_test_A = dfa_test_A.drop(columns=['Doughnuts consumption'])

print("Number of NaNs in each column after processing: ")
print(dfa_train_A.isnull().sum())

print("dfa_train_A after processing: ")
display(dfa_train_A)

#imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#imp.fit(X)
#X = imp.transform(X)

data = np.asarray(dfa_train_A)

# Let's remove the first column, which is the ID
X = data[:,1:-1]
#X = data[:,:-1]
print("Original X: ")
print(X)
# Let's convert part of the training data into test data, so that we can test the imputer
X_train = X[size_validation:]
X_test = X[:size_validation]

# The last column is the target: is fake or real, but we convert it into 0 (fake) and 1 (real)
y = data[:,-1]
y = np.where(y == 'fake', 0, 1)
print("y: ")
print(y)

# The true values were
y_train = y[size_validation:]
y_test = y[:size_validation]

# Now let's test the imputer
print("--------------------")
print("DecisionTreeClassifier")
print("--------------------")
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print("predictions", predictions)
print("true_values: ", y_test)

print("Score of the method: ")
print(metrics.accuracy_score(y_test, predictions))

plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=dfa_train_A.columns[1:-1], class_names=True)
plt.show()

# Now use the whole training data to train the model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
feature_importances = clf.feature_importances_
print(dict(zip(dfa_train_A.columns[1:], feature_importances)))

# And use the test data to predict the values
data_test = np.asarray(dfa_test_A)
X_test = data_test[:,1:]
predictions = clf.predict(X_test)
print("predictions: ")
print(predictions)

# Convert it back to 'real' and 'fake'
predictions = np.where(predictions == 0, 'fake', 'real')

# Save the predictions
df_predictions = pd.DataFrame(predictions)
df_predictions.columns = ['Prediction']
df_predictions.to_csv("predictions_1.csv", index=False, header=True)

exit(0)








# Let's try another method
print("--------------------")
print("DecisionTreeRegressor")
print("--------------------")

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("predictions: ")
print(predictions)
print("true_values: ")
print(y_test)
print(metrics.accuracy_score(y_test, predictions))