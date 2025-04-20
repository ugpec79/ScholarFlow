# userrecom/engine.py

import pandas as pd
import numpy as np
import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
from collections import defaultdict
import ast
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def generate_user_features():
    # Load articles
    articles = pd.read_csv("cleaned_data.csv")
    articles = articles.dropna(subset=["title", "abstract"])
    articles["fos"] = articles["fos"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    articles["keywords"] = articles["keywords"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    # Venue popularity
    venue_popularity = articles["venue"].value_counts().to_dict()
    articles["venue_popularity"] = articles["venue"].map(
        lambda v: venue_popularity.get(v, 0)
    )

    # Load user interactions
    def load_user_interactions(path="user_actions.csv"):
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return pd.DataFrame(rows)

    interactions = load_user_interactions()

    # Initial action weights (base labels before sentiment boost)
    action_weights = {"LIKE": 2, "CLICKED": 1, "COMMENT": 0, "DISLIKE": 0}
    interactions["label"] = interactions["action"].map(action_weights).astype(float)

    # Perform real sentiment analysis on comments
    print("🔍 Performing batched sentiment analysis using DistilBERT...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    interactions["sentiment_score"] = 0.0

    comments = interactions["comment"].fillna("").tolist()
    results = sentiment_pipeline(comments, batch_size=32, truncation=True)

    for i, res in enumerate(results):
        score = res["score"]
        sentiment = 1.0 if res["label"] == "POSITIVE" else -1.0
        interactions.at[i, "sentiment_score"] = score * sentiment

        # Apply sentiment impact only for comments
        if interactions.at[i, "action"] == "COMMENT" and sentiment > 0:
            interactions.at[i, "label"] = max(interactions.at[i, "label"], 1.0)

    # Aggregate by (user, paper)
    agg_interactions = (
        interactions.groupby(["user_id", "paper_id"])
        .agg({"label": "max", "sentiment_score": "mean"})
        .reset_index()
    )

    # Load and embed articles
    print("💡 Encoding article text embeddings...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    articles["content"] = articles["title"] + ". " + articles["abstract"]
    articles["embedding"] = list(
        model.encode(articles["content"].tolist(), show_progress_bar=True)
    )

    # Merge user interactions with article metadata
    merged = agg_interactions.merge(
        articles, left_on="paper_id", right_on="_id", how="inner"
    )

    # Build user profiles: embeddings, FOS, and keywords
    user_profiles = {}
    user_fos = defaultdict(set)
    user_keywords = defaultdict(set)

    for user_id, group in merged.groupby("user_id"):
        liked_articles = group[group["label"] >= 1.0]

        if not liked_articles.empty:
            embeds = np.stack(liked_articles["embedding"].values)
            user_profiles[user_id] = np.mean(embeds, axis=0)
        else:
            user_profiles[user_id] = np.zeros(384)

        for _, row in liked_articles.iterrows():
            user_fos[user_id].update(row["fos"])
            user_keywords[user_id].update(row["keywords"])

    # Compute personalized similarity features
    def compute_similarity(row):
        user_vector = user_profiles.get(row["user_id"], np.zeros(384))
        return cosine_similarity([user_vector], [row["embedding"]])[0][0]

    def compute_fos_overlap(row):
        user_set = user_fos.get(row["user_id"], set())
        return len(user_set.intersection(set(row["fos"]))) / (
            len(user_set.union(set(row["fos"]))) + 1e-5
        )

    def compute_keyword_match(row):
        user_kw = user_keywords.get(row["user_id"], set())
        paper_kw = set(row["keywords"])
        return len(user_kw.intersection(paper_kw)) / (
            len(user_kw.union(paper_kw)) + 1e-5
        )

    print("🔧 Computing feature similarity metrics...")
    merged["content_sim"] = merged.apply(compute_similarity, axis=1)
    merged["fos_overlap"] = merged.apply(compute_fos_overlap, axis=1)
    merged["keyword_match"] = merged.apply(compute_keyword_match, axis=1)

    # Final feature table
    features = pd.DataFrame(
        {
            "user_id": merged["user_id"],
            "article_id": merged["_id"],
            "label": merged["label"],
            "sentiment_score": merged["sentiment_score"],
            "n_citation": merged["n_citation"].astype(float),
            "year": merged["year"].astype(int),
            "venue_popularity": merged["venue_popularity"],
            "content_sim": merged["content_sim"],
            "fos_overlap": merged["fos_overlap"],
            "keyword_match": merged["keyword_match"],
        }
    )

    # Encode IDs for ranking model
    le_user = LabelEncoder()
    features["user_id_enc"] = le_user.fit_transform(features["user_id"])
    le_item = LabelEncoder()
    features["article_id_enc"] = le_item.fit_transform(features["article_id"])

    # Save final feature set
    features.to_csv("data/training_features.csv", index=False)
    print("✅ Feature set saved with full interaction and similarity logic.")


def train_ranking_model():
    # Load Features
    data = pd.read_csv("data/training_features.csv")

    # Feature & Target Columns
    feature_cols = [
        "sentiment_score",
        "n_citation",
        "year",
        "venue_popularity",
        "content_sim",
        "fos_overlap",
        "keyword_match",
    ]
    X = data[feature_cols]
    y = data["label"]
    group = data.groupby("user_id_enc").size().tolist()

    # Split Data for Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.01, random_state=42
    )

    # Group info (re-calculated for split)
    user_ids = data["user_id_enc"].values
    train_user_ids = user_ids[: len(X_train)]
    val_user_ids = user_ids[len(X_train) :]

    train_group = pd.Series(train_user_ids).value_counts().sort_index().tolist()
    val_group = pd.Series(val_user_ids).value_counts().sort_index().tolist()

    # LightGBM Datasets
    train_data = lgb.Dataset(X_train, label=y_train, group=train_group)
    val_data = lgb.Dataset(X_val, label=y_val, group=val_group)

    # Parameters for LambdaRank
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "ndcg_eval_at": [5],
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "early_stopping_round": 300,
    }

    # Train Model
    print("🚀 Training LightGBM LambdaRank model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=300,
    )

    # Save Model
    os.makedirs("models", exist_ok=True)
    model.save_model("models/ltr_model.txt")
    print("✅ Model trained and saved at models/ltr_model.txt")
