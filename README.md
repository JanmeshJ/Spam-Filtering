1. Prerequisites
First, we’ll import the necessary dependencies. Pandas is a library used mostly used by data scientists for data cleaning and analysis.

Scikit-learn, also called Sklearn, is a robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling, including classification, regression, clustering, and dimensionality reduction via a consistent interface.

Run the command below to import the necessary dependencies:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

2. Getting started
To get started, first, run the code below:

spam = pd.read_csv('spam.csv')

In the code above, we created a spam.csv file, which we’ll turn into a data frame and save to our folder spam. A data frame is a structure that aligns data in a tabular fashion in rows and columns.

3. Python train_test_split()
We’ll use a train-test split method to train our email spam detector to recognize and categorize spam emails. The train-test split is a technique for evaluating the performance of a machine learning algorithm. We can use it for either classification or regression of any supervised learning algorithm.

The procedure involves taking a dataset and dividing it into two separate datasets. The first dataset is used to fit the model and is referred to as the training dataset. For the second dataset, the test dataset, we provide the input element to the model. Finally, we make predictions, comparing them against the actual output.

Train dataset: used to fit the machine learning model
Test dataset: used to evaluate the fit of the machine learning model

In practice, we’d fit the model on available data with known inputs and outputs. Then, we’d make predictions based on new examples for which we don’t have the expected output or target values. We’ll take the data from our sample .csv file, which contains examples pre-classified into spam and non-spam, using the labels spam and ham, respectively.

4. Extracting features
Next, we’ll run the code below:
cv = CountVectorizer()
features = cv.fit_transform(z_train)
In cv= CountVectorizer(), CountVectorizer() randomly assigns a number to each word in a process called tokenizing. Then, it counts the number of occurrences of words and saves it to cv. At this point, we’ve only assigned a method to cv.

features = cv.fit_transform(z_train) randomly assigns a number to each word. It counts the number of occurrences of each word, then saves it to cv. In the image below, 0 represents the index of the email. The number sequences in the middle column represent a word recognized by our function, and the numbers on the right indicate the number of times that word was counted

5. Testing our email spam detector
ow, to ensure accuracy, let’s test our application. Run the code below:

features_test = cv.transform(z_test)
print("Accuracy: {}".format(model.score(features_test,y_test)))
The features_test = cv.transform(z_test) function makes predictions from z_test that will go through count vectorization. It saves the results to the features_test file.

In the print(model.score(features_test,y_test)) function, mode.score() scores the prediction of features_test against the actual labels in y_test.

In the project above, you’ll see that we were able to classify spam with 97 percent accuracy.
