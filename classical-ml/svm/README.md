# Support Vector Machines
Examples using support vector machines for classification and regression.

## Examples
### Linear SVC - Spam Classification

Classifies messages as spam or not spam using TF-IDF features.

- **Task:** Binary classification
- **Model:** Linear SVC
- **Input:** Message text transformed into TF-IDF features
- **Output:** Spam or not spam
- **File:** `linear-svc-spam-classification.ipynb`

### SVR — House Price Regression
Predicts house prices using house size and number of bedrooms.

- **Task:** Regression
- **Model:** Support vector regression
- **Input:** House size and number of bedrooms
- **Output:** Predicted house price
- **File:** `svr-house-price-regression.ipynb`

## Key concepts
- **SVC:** Learns a decision boundary for classification.
- **SVR:** Predicts continuous values using support vector regression.
- **C:** Controls the strength of regularization.
- **Kernel:** Determines the type of relationship the model can learn.
- **Feature scaling:** Often important for numerical SVM inputs.