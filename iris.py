import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression

def train_iris_classifier():
    # Load Iris dataset
    iris = datasets.load_iris()
    
    # Create DataFrame with column names
    df = pd.DataFrame(iris['data'], columns=['sl', 'sw', 'pl', 'pw'])
    df['target'] = iris['target']

    # Split into train and test sets
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

    # Separate features and targets
    X_train = df_train.drop(columns='target').values
    Y_train = df_train['target'].values
    X_test = df_test.drop(columns='target').values
    Y_test = df_test['target'].values

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)

    # Accuracy metrics
    test_score = model.score(X_test, Y_test)
    cross_val_acc = np.mean(cross_val_score(model, X_train, Y_train, cv=5))

    # Add predictions to the training set
    Y_pred = cross_val_predict(model, X_train, Y_train, cv=5)
    df_train['prediction'] = Y_pred
    df_train['correct'] = (Y_pred == Y_train)

    def predict(new_data):
        """
        Predict target class for new input.
        new_data: list or np.array of shape (n_samples, 4)
        returns: predicted class labels
        """
        new_data = np.array(new_data)
        return model.predict(new_data)

    return {
        'model': model,
        'train_score': test_score,
        'cross_val_accuracy': cross_val_acc,
        'train_data': df_train,
        'predict': predict
    }
classifier = train_iris_classifier()

# Print scores
print("Test Accuracy:", classifier['train_score'])
print("Cross-Validation Accuracy:", classifier['cross_val_accuracy'])

# Predict on new data
new_samples  = [[5.1, 3.5, 1.4, 0.2],   # likely setosa
                [6.2, 3.4, 5.4, 2.3]]   # likely virginica
predictions = classifier['predict'](new_samples)
#hello
print("Predictions:", predictions)
