import os
from .engine import generate_user_features
from .engine import train_ranking_model

def retrain_model_pipeline():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("📊 Generating user features...")
    generate_user_features()

    print("🧠 Training ranking model...")
    train_ranking_model()

    print("✅ Retraining pipeline completed.")
