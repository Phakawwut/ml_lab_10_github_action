import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import os


def preprocess_titanic_data(test_size=0.2, random_state=42):
    """
    Preprocesses Titanic data with feature engineering
    """
    mlflow.set_experiment("Titanic - Data Preprocessing")
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"🔄 Starting preprocessing with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")
        
        # Load data
        df = pd.read_csv("data/titanic.csv")
        print(f"📊 Original data shape: {df.shape}")
        
        # Feature Engineering
        print("🛠️ Performing feature engineering...")
        
        # 1. Handle missing values
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        
        # 2. Create new features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'
        }
        df['Title'] = df['Title'].map(title_mapping).fillna('Other')
        
        # 3. Encode categorical variables
        le = LabelEncoder()
        df['Sex_encoded'] = le.fit_transform(df['Sex'])
        df['Embarked_encoded'] = le.fit_transform(df['Embarked'])
        df['Title_encoded'] = le.fit_transform(df['Title'])
        
        # 4. Create age bins
        df['AgeBin'] = pd.cut(df['Age'], bins=5, labels=[0,1,2,3,4])
        df['FareBin'] = pd.cut(df['Fare'], bins=4, labels=[0,1,2,3])
        
        # 5. Select features for modeling
        feature_columns = [
            'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_encoded', 'FamilySize', 'IsAlone', 'Title_encoded',
            'AgeBin', 'FareBin'
        ]
        
        # Prepare final dataset
        X = df[feature_columns]
        y = df['Survived']
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        print(f"🎯 Final feature set shape: {X.shape}")
        print(f"📋 Features: {feature_columns}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Save processed data
        processed_dir = "processed_data"
        os.makedirs(processed_dir, exist_ok=True)
        
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv(f"{processed_dir}/train.csv", index=False)
        test_data.to_csv(f"{processed_dir}/test.csv", index=False)
        
        # Log parameters and metrics
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("num_features", len(feature_columns))
        mlflow.log_param("feature_list", feature_columns)
        
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("train_survival_rate", y_train.mean())
        mlflow.log_metric("test_survival_rate", y_test.mean())
        
        # Log artifacts
        mlflow.log_artifacts(processed_dir, artifact_path="processed_data")
        
        print(f"✅ Preprocessing completed!")
        print(f"📁 Training samples: {len(X_train)}")
        print(f"📁 Test samples: {len(X_test)}")
        print(f"🎯 Training survival rate: {y_train.mean():.3f}")
        print(f"🎯 Test survival rate: {y_test.mean():.3f}")
        
        print("-" * 50)
        print(f"🔑 Preprocessing Run ID: {run_id}")
        print("-" * 50)


# Write run_id to GITHUB_OUTPUT for other steps in the workflow to use
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={run_id}", file=f)


if __name__ == "__main__":
    # This line correctly calls the function to ensure the script runs
    preprocess_titanic_data()

