from sklearn.datasets import load_iris
import mlflow


def load_and_predict():
    """
    Simulates a production scenario by loading a model from a specific
    stage in the MLflow Model Registry and using it for prediction.
    """
    MODEL_NAME = "iris-classifier-prod"
    MODEL_STAGE = "Staging"  # Change to "Production" after transitioning the model stage

    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")

    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version is in the '{MODEL_STAGE}' stage in the MLflow UI.")
        return

    X, y = load_iris(return_X_y=True, as_frame=False)
    sample_data = X[0:1]  # Use the first row as a sample
    actual_label = y[0]

    prediction = model.predict(sample_data)

    print("-" * 30)
    print(f"Sample Data Features:\n{sample_data[0]}")
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {prediction[0]}")
    print("-" * 30)


if __name__ == "__main__":
    load_and_predict()
