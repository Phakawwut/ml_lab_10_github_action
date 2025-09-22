import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts


def train_multiple_models(preprocessing_run_id):
    """
    Trains multiple models and registers the best one
    """
    ACCURACY_THRESHOLD = 0.80
    mlflow.set_experiment("Titanic - Model Training")
    
    # Load processed data
    try:
        local_artifact_path = download_artifacts(
            run_id=preprocessing_run_id,
            artifact_path="processed_data"
        )
        
        train_df = pd.read_csv(os.path.join(local_artifact_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(local_artifact_path, "test.csv"))
        print("✅ Successfully loaded processed data")
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        sys.exit(1)
    
    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']
    X_test = test_df.drop('Survived', axis=1)
    y_test = test_df['Survived']
    
    # Models to train
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'RandomForest': Pipeline([
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    for model_name, pipeline in models.items():
        with mlflow.start_run(run_name=f"titanic_{model_name.lower()}"):
            print(f"🚀 Training {model_name}...")
            mlflow.set_tag("ml.step", "model_training")
            mlflow.set_tag("model_type", model_name)
            mlflow.log_param("preprocessing_run_id", preprocessing_run_id)
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc", auc)
            
            # Log model
            mlflow.sklearn.log_model(pipeline, f"titanic_{model_name.lower()}")
            
            print(f"📊 {model_name} Results:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   AUC: {auc:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = pipeline
                best_model_name = model_name
                best_run_id = mlflow.active_run().info.run_id
    
    # Register best model
    if best_accuracy >= ACCURACY_THRESHOLD:
        print(f"\n🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        print("📝 Registering model...")
        
        model_uri = f"runs:/{best_run_id}/titanic_{best_model_name.lower()}"
        registered_model = mlflow.register_model(model_uri, "titanic-survival-predictor")
        
        print(f"✅ Model registered as '{registered_model.name}' version {registered_model.version}")
    else:
        print(f"❌ Best accuracy ({best_accuracy:.4f}) below threshold ({ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id>")
        sys.exit(1)
    
    run_id = sys.argv[1]
    train_multiple_models(preprocessing_run_id=run_id)
