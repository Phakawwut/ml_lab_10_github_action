import pandas as pd
import mlflow
import os


def validate_titanic_data():
    """
    Validates the Titanic dataset and logs results to MLflow
    """
    mlflow.set_experiment("Titanic - Data Validation")
    
    with mlflow.start_run():
        print("🔍 Starting Titanic data validation...")
        mlflow.set_tag("ml.step", "data_validation")
        mlflow.set_tag("dataset", "titanic")
        
        # Load data
        data_path = "data/titanic.csv"
        if not os.path.exists(data_path):
            print("❌ Error: titanic.csv not found in data/ folder")
            return
            
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Basic validation
        num_rows, num_cols = df.shape
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (num_rows * num_cols)) * 100
        
        # Check target distribution
        if 'Survived' in df.columns:
            target_distribution = df['Survived'].value_counts().to_dict()
            survival_rate = df['Survived'].mean()
        else:
            print("❌ Error: 'Survived' column not found!")
            return
        
        # Log metrics
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric("missing_percentage", missing_percentage)
        mlflow.log_metric("survival_rate", survival_rate)
        
        # Log parameters
        mlflow.log_param("dataset_shape", f"{num_rows}x{num_cols}")
        mlflow.log_param("target_distribution", target_distribution)
        
        # Validation checks
        validation_status = "SUCCESS"
        if missing_percentage > 50:
            validation_status = "FAILED - Too many missing values"
        elif survival_rate < 0.1 or survival_rate > 0.9:
            validation_status = "WARNING - Imbalanced target"
            
        mlflow.log_param("validation_status", validation_status)
        
        # Print summary
        print("\n📊 Data Validation Summary:")
        print(f"   Dataset Shape: {num_rows} rows × {num_cols} columns")
        print(f"   Missing Values: {missing_values} ({missing_percentage:.2f}%)")
        print(f"   Survival Rate: {survival_rate:.3f}")
        print(f"   Target Distribution: {target_distribution}")
        print(f"   Status: {validation_status}")
        
        # Column info
        print(f"\n📋 Columns: {list(df.columns)}")
        print(f"📋 Data Types:\n{df.dtypes}")
        
        print("✅ Data validation completed!")


if __name__ == "__main__":
    validate_titanic_data()