import mlflow

# Start a new MLflow run
mlflow.start_run()

# Log a parameter and a metric
mlflow.log_param("my", "param")
mlflow.log_metric("score", 100)

# End the run
mlflow.end_run()

# Alternatively, using the context manager (preferred)
with mlflow.start_run() as run:
    mlflow.log_param("my", "param")
    mlflow.log_metric("score", 100)
