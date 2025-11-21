Downlaod your diabetes dataset from sources dont just copy paste else you will get nothing to understand the working of the program


1)Importing the Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

{
pandas: Handles data like tables in Excel, lets you read, organize, and analyze information (your CSV dataset).​
numpy: Helps perform math operations, especially with large sets of numbers.
StandardScaler: Makes sure different features (like age, glucose, BMI) are on a similar scale, which helps machine learning algorithms work better.
train_test_split: Breaks your dataset into two parts: one to teach the computer (training), and one to test it (testing).
svm: Stands for Support Vector Machine—a kind of algorithm that can classify items (in this case, predicts diabetic/non-diabetic).
accuracy_score: Tells you how many times your predictions were correct.
}

2)
diabetes_dataset = pd.read_csv('path_to_your_file/diabetes.csv')

Reads your diabetes data from a CSV file into a pandas dataframe

3)Previewing the Dataset

print(diabetes_dataset.head())
print(diabetes_dataset.shape)
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())
print(diabetes_dataset.groupby('Outcome').mean())

{
    head(): Shows first five rows—useful to check if the data loaded correctly.
shape: Tells you number of rows and columns.
describe(): Gives average, minimum, maximum, etc. for each attribute.
value_counts(): Shows how many diabetic (1) and not diabetic (0) cases.
groupby('Outcome').mean(): Shows the average values of each factor for diabetic vs non-diabetic.
}

4)Separating Features and Labels

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

5)Data Standardization

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
print(X)
print(Y)

Machine learning works best if input features are similar in scale—StandardScaler does this by subtracting the mean and dividing by the standard deviation for each column.
After scaling, you use these values for training and prediction

6)Splitting the Data

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

Splits your dataset into training (80%) and testing (20%) sets.
train: Used to teach the computer.
test: Used to see how well it learned to predict diabetes.
stratify=Y keeps the ratio of diabetic/non-diabetic similar in both sets.​


7)Training the SVM Model

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

Creates a machine learning model using Support Vector Machine (SVM).

kernel='linear': Looks for straight lines to separate classes.

fit(): Actually trains the model with your training data.


8)Checking Accuracy

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of the training data:', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of the test data:', test_data_accuracy)

Predicts the outcome for the training and test sets.

Compares predicted vs. actual results; accuracy_score reports how often the predictions were correct.

Print statements show how well your machine learning model works.

9)Prediction on New Data

input_data = (10, 168, 74, 0, 0, 38, 0.537, 34)
input_data_as_numpy_array = np.array(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)

{if (prediction == 0):
    print("The person is not diabetic.")
else:
    print("The person is diabetic.")

Create a tuple with health parameters for a new person.

Convert it to a numpy array and reshape it so the model understands it’s one sample.

Standardize this new data (just like original data).

The model predicts: 0 for not diabetic, 1 for diabetic.

Prints friendly message with the result.

}
