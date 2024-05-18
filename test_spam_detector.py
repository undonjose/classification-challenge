import re

def search_file(file, pattern):
    content = open(file).read()
    if re.search(pattern, content, re.MULTILINE | re.DOTALL):
        return True
    return False

### Split the Data into Training and Testing Sets (30 points)
def test_split():
    assert search_file("spam_detector.ipynb", rf"Based on accuracy of prior practice activities, I'd expect Random Forest Classifier to have a higher accuracy.*Random Forest is better able to handle nonlinear decision boundaries.") == True, "There is a prediction about which model you expect to do better. (5 points)"

    assert search_file("spam_detector.ipynb", rf"y = data\[\'spam\'\]") == True, "The labels set (y) is created from the “spam” column. (5 points)"

    assert search_file("spam_detector.ipynb", rf"X = data.drop\(\'spam\', axis=\'columns\'\)") == True, "The features DataFrame (X) is created from the remaining columns. (5 points)"

    assert search_file("spam_detector.ipynb", rf"y\.value_counts\(\)") == True, "The value_counts function is used to check the balance of the labels variable (y). (5 points)"

    assert search_file("spam_detector.ipynb", rf"X_train, X_test, y_train, y_test = train_test_split\(X, y\)") == True, "The data is correctly split into training and testing datasets by using train_test_split. (10 points)"

### Scale the Features (20 points)
def test_scale():
    assert search_file("spam_detector.ipynb", rf"scaler = StandardScaler\(\)") == True, "An instance of StandardScaler is created. (5 points)"

    assert search_file("spam_detector.ipynb", rf"scaler = scaler.fit\(X_train\)") == True, "The Standard Scaler instance is fit with the training data. (5 points)"

    assert search_file("spam_detector.ipynb", rf"X_train_scaled = scaler\.transform\(X_train\)") == True, "The training features DataFrame is scaled using the transform function. (5 points)"

    assert search_file("spam_detector.ipynb", rf"X_test_scaled = scaler\.transform\(X_test\)") == True, "The testing features DataFrame is scaled using the transform function. (5 points)"

### Create a Logistic Regression Model (20 points)
def test_lr_model():
    assert search_file("spam_detector.ipynb", rf"lr_model = LogisticRegression\(random_state=1\)") == True, "A logistic regression model is created with a random_state of 1. (5 points)"

    assert search_file("spam_detector.ipynb", rf"lr_model.fit\(X_train_scaled, y_train\)") == True, "The logistic regression model is fitted to the scaled training data (X_train_scaled and y_train). (5 points)"

    assert search_file("spam_detector.ipynb", rf"lr_testing_predictions = lr_model\.predict\(X_test_scaled\)") == True, "Predictions are made for the testing data labels by using the testing feature data (X_test_scaled) and the fitted model, and saved to a variable. (5 points)"

    assert search_file("spam_detector.ipynb", rf"accuracy_score\(y_test, lr_testing_predictions\)") == True, "The model’s performance is evaluated by calculating the accuracy score of the model with the accuracy_score function. (5 points)"

### Create a Random Forest Model (20 points)
def test_rf_model():
    assert search_file("spam_detector.ipynb", rf"rfc_model = RandomForestClassifier\(random_state=1\)") == True, "A random forest model is created with a random_state of 1. (5 points)"

    assert search_file("spam_detector.ipynb", rf"rfc_model.fit\(X_train_scaled, y_train\)") == True, "The random forest model is fitted to the scaled training data (X_train_scaled and y_train). (5 points)"

    assert search_file("spam_detector.ipynb", rf"rfc_testing_predictions = rfc_model\.predict\(X_test_scaled\)") == True, "Predictions are made for the testing data labels by using the testing feature data (X_test_scaled) and the fitted model, and saved to a variable. (5 points)"

    assert search_file("spam_detector.ipynb", rf"accuracy_score\(y_test, rfc_testing_predictions\)") == True, "The model’s performance is evaluated by calculating the accuracy score of the model with the accuracy_score function. (5 points)"

### Evaluate the Models (10 points)
def test_eval():
    assert search_file("spam_detector.ipynb", rf"With Logistic Regression scoring around 0\.922 accuracy and Random Forest Classifier scoring about 0\.960, Random Forest was more accurate") == True, "Which model performed better? (5 points)"

    assert search_file("spam_detector.ipynb", rf", which matched my prediction") == True, "How does that compare to your prediction? (5 points)"
