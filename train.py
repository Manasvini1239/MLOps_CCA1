import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Create/select experiment
mlflow.set_experiment("Iris_ML_Experiment")


# Start MLflow run
with mlflow.start_run():

    # Load Iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create the model
    model = LogisticRegression(max_iter=200)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters in MLflow
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("max_iter", 200)

    # Log accuracy in MLflow
    mlflow.log_metric("accuracy", accuracy)

    # Print results
    print("Model trained successfully!")
    print("Accuracy:", accuracy)
    print("MLflow tracking completed!")