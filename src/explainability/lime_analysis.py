import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer

class AQILimeAnalyzer:
    def __init__(self, model_path="models/registry/registry/24h_model.joblib", 
                 train_data_path="data/training/features_train.parquet",
                 test_data_path="data/training/features_test.parquet"):
        self.model_path = Path(model_path)
        self.train_data_path = Path(train_data_path)
        self.test_data_path = Path(test_data_path)
        self.model = None
        self.X_train_sample = None
        self.X_test = None
        self.feature_names = None
        self.explainer = None

    def load_assets(self, background_size=300):
        """Loads the model and datasets, isolating the 626 features[cite: 1]."""
        print(f"📦 Loading model from {self.model_path}[cite: 1]...")
        self.model = joblib.load(self.model_path)
        
        print(f"📊 Loading training and test data splits[cite: 1]...")
        train_df = pd.read_parquet(self.train_data_path)
        test_df = pd.read_parquet(self.test_data_path)
        
        # Drop metadata and target columns to match the 626 feature vector
        drop_cols = [
            'timestamp', 'city', 'country', 'created_at', 'source', 
            'aqi_category', 'dominant_pollutant', 
            'target_aqi_t+24', 'target_aqi_t+48', 'target_aqi_t+72', 'aqi'
        ]
        
        self.feature_names = [c for c in train_df.columns if c not in drop_cols]
        
        # Take a sample from the training set for LIME's background distribution
        self.X_train_sample = train_df[self.feature_names].sample(
            n=min(background_size, len(train_df)), random_state=42
        ).values
        
        self.X_test = test_df[self.feature_names]
        print(f"🎯 Feature matrix ready with {len(self.feature_names)} features[cite: 1].")

    def initialize_explainer(self):
        """Initializes the LIME Tabular Explainer for regression tasks."""
        print("⚙️ Initializing LIME Tabular Explainer...")
        self.explainer = LimeTabularExplainer(
            training_data=self.X_train_sample,
            feature_names=self.feature_names,
            class_names=['predicted_aqi'],
            mode='regression',
            random_state=42
        )

    def explain_single_instance(self, instance_idx=0, save_path="models/registry/registry/evaluation/lime_explanation.html"):
        """Explains a specific test instance prediction and saves it as an interactive HTML file."""
        instance = self.X_test.iloc[instance_idx].values
        
        # Define prediction function compatible with LIME
        predict_fn = lambda x: self.model.predict(x)
        
        print(f"🔍 Generating LIME explanation for test instance index {instance_idx}...")
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=10
        )
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        exp.save_to_file(save_path)
        print(f"✅ LIME explanation successfully saved to {save_path}")
        return exp

if __name__ == "__main__":
    # Run standalone test
    analyzer = AQILimeAnalyzer()
    analyzer.load_assets(background_size=300)
    analyzer.initialize_explainer()
    analyzer.explain_single_instance(instance_idx=0)