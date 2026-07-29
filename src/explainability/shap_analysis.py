import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

class AQIShapAnalyzer:
    def __init__(self, model_path="models/registry/registry/24h_model.joblib", 
                 data_path="data/training/features_test.parquet"):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.df = None
        self.X = None
        self.explainer = None
        self.shap_values = None

    def load_assets(self, sample_size=300):
        """Loads the champion model and test dataset, extracting the features."""
        print(f"📦 Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        
        print(f"📊 Loading dataset from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        
        # Drop metadata and target columns to isolate features
        drop_cols = [
            'timestamp', 'city', 'country', 'created_at', 'source', 
            'aqi_category', 'dominant_pollutant', 
            'target_aqi_t+24', 'target_aqi_t+48', 'target_aqi_t+72', 'aqi'
        ]
        feature_cols = [c for c in self.df.columns if c not in drop_cols]
        
        # Take a random sample for faster SHAP calculation
        self.X = self.df[feature_cols].sample(n=min(sample_size, len(self.df)), random_state=42)
        print(f"🎯 Feature matrix shape for SHAP analysis: {self.X.shape}")

    def compute_shap_values(self):
        """Computes SHAP values using TreeExplainer optimized for tree-based models."""
        print("⚙️ Initializing SHAP TreeExplainer & calculating values...")
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(self.X)
        print("✅ SHAP values successfully calculated!")
        return self.shap_values

    def plot_summary(self, save_path="models/registry/registry/evaluation/shap_summary.png"):
        """Generates and saves the global SHAP summary plot."""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, self.X, show=False)
        plt.title("SHAP Global Feature Importance - AQI Forecasting Model", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP summary plot successfully saved to {save_path}")

    def get_top_features(self, top_n=10):
        """Extracts top N most important features based on mean absolute SHAP value."""
        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values(by='mean_abs_shap', ascending=False)
        
        return feature_importance.head(top_n)

if __name__ == "__main__":
    analyzer = AQIShapAnalyzer()
    analyzer.load_assets(sample_size=300)
    analyzer.compute_shap_values()
    analyzer.plot_summary()
    
    top_feats = analyzer.get_top_features(10)
    print("\n🏆 Top 10 Most Impactful Features (SHAP):")
    print(top_feats.to_string(index=False))