"""
Genre Classification Experiment
Predicts an artist's main_genre from their followers, popularity, and niche genres.
Tracks experiments to DagsHub via MLflow.

Setup:
    pip install dagshub mlflow scikit-learn pandas

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import dagshub
import ast
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DAGSHUB SETUP - This connects MLflow to your DagsHub repo
# ============================================================
dagshub.init(repo_owner='amccarty', repo_name='my-first-repo', mlflow=True)

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


def load_and_prepare_data(filepath='artists.csv'):
    """Load artist data and prepare features for classification."""
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} artists")
    
    # Parse niche genres from string representation of list
    def parse_genres(genres_str):
        try:
            if pd.isna(genres_str) or genres_str == '[]':
                return []
            return ast.literal_eval(genres_str)
        except:
            return []
    
    df['niche_genres_list'] = df['genres'].apply(parse_genres)
    
    # Remove artists with no main_genre
    df = df.dropna(subset=['main_genre'])
    
    # Filter to main genres with enough samples
    genre_counts = df['main_genre'].value_counts()
    valid_genres = genre_counts[genre_counts >= 100].index.tolist()
    df = df[df['main_genre'].isin(valid_genres)]
    
    print(f"After filtering: {len(df):,} artists across {len(valid_genres)} genres")
    print(f"Genres: {valid_genres}")
    
    return df


def create_features(df):
    """Create feature matrix from artist data."""
    
    # Numeric features
    X_numeric = df[['followers', 'popularity']].copy()
    
    # Log transform followers (very skewed)
    X_numeric['log_followers'] = np.log1p(X_numeric['followers'])
    X_numeric = X_numeric.drop('followers', axis=1)
    
    # Binarize niche genres (multi-hot encoding)
    mlb = MultiLabelBinarizer(sparse_output=False)
    niche_encoded = mlb.fit_transform(df['niche_genres_list'])
    
    # Only keep niche genres that appear at least 50 times
    genre_counts = niche_encoded.sum(axis=0)
    common_genres_mask = genre_counts >= 50
    niche_encoded = niche_encoded[:, common_genres_mask]
    niche_genre_names = [g for g, keep in zip(mlb.classes_, common_genres_mask) if keep]
    
    X_niche = pd.DataFrame(niche_encoded, columns=niche_genre_names, index=df.index)
    
    # Combine features
    X = pd.concat([X_numeric.reset_index(drop=True), X_niche.reset_index(drop=True)], axis=1)
    
    # Target
    le = LabelEncoder()
    y = le.fit_transform(df['main_genre'])
    
    print(f"Feature matrix: {X.shape}")
    print(f"  - Numeric features: {X_numeric.shape[1]}")
    print(f"  - Niche genre features: {X_niche.shape[1]}")
    
    return X, y, le


def run_experiment(X, y, label_encoder, model_type='random_forest', **model_params):
    """Run a single experiment and log to MLflow."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Select model
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, **model_params)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42, **model_params)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", len(y))
        mlflow.log_param("n_classes", len(label_encoder.classes_))
        mlflow.log_param("test_size", 0.2)
        
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        # Train
        print(f"\nTraining {model_type} with params: {model_params}")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log classification report as artifact
        report = classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=False
        )
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        
        # Feature importance for tree models
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
            print(f"\n  Top 10 features:")
            for _, row in importance_df.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
        
        return accuracy, f1_macro


def main():
    print("=" * 60)
    print("Genre Classification Experiment")
    print("Tracking to DagsHub MLflow")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data('artists.csv')
    
    # Create features
    X, y, label_encoder = create_features(df)
    
    # Run multiple experiments
    experiments = [
        # Random Forest experiments
        ('random_forest', {'n_estimators': 50, 'max_depth': 10}),
        ('random_forest', {'n_estimators': 100, 'max_depth': 15}),
        ('random_forest', {'n_estimators': 200, 'max_depth': 20}),
        
        # Gradient Boosting experiments
        ('gradient_boosting', {'n_estimators': 50, 'max_depth': 5}),
        ('gradient_boosting', {'n_estimators': 100, 'max_depth': 7}),
        
        # Logistic Regression
        ('logistic_regression', {'C': 0.1}),
        ('logistic_regression', {'C': 1.0}),
        ('logistic_regression', {'C': 10.0}),
    ]
    
    results = []
    for model_type, params in experiments:
        accuracy, f1 = run_experiment(X, y, label_encoder, model_type, **params)
        results.append({
            'model': model_type,
            'params': params,
            'accuracy': accuracy,
            'f1_macro': f1
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    results_df = pd.DataFrame(results).sort_values('f1_macro', ascending=False)
    print(results_df.to_string(index=False))
    
    print(f"\nView experiments at: https://dagshub.com/amccarty/my-first-repo/experiments")


if __name__ == "__main__":
    main()
