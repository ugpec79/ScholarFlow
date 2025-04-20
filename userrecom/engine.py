# ======================= engine.py ===========================
from lightfm import LightFM
from lightfm.data import Dataset as LFMData
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
from collections import defaultdict
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd
import numpy as np
import csv, ast, os


def load_user_interactions(path="user_actions.csv"):
    with open(path, newline='') as f:
        return pd.DataFrame(list(csv.DictReader(f)))


def generate_latent_embeddings(interactions):
    print("🎯 Training LightFM matrix factorization model...")

    lfm_data = LFMData()
    lfm_data.fit(interactions["user_id"].unique(), interactions["paper_id"].unique())
    tuples = list(zip(interactions["user_id"], interactions["paper_id"]))
    matrix, _ = lfm_data.build_interactions(tuples)

    model = LightFM(no_components=32, loss='warp')
    model.fit(matrix, epochs=10, num_threads=4)

    user_id_map, _, item_id_map, _ = lfm_data.mapping()
    rev_user_map = {v: k for k, v in user_id_map.items()}
    rev_item_map = {v: k for k, v in item_id_map.items()}

    user_embeds = pd.DataFrame(model.user_embeddings)
    user_embeds["user_id"] = user_embeds.index.map(rev_user_map)

    item_embeds = pd.DataFrame(model.item_embeddings)
    item_embeds["article_id"] = item_embeds.index.map(rev_item_map)

    print("📦 LightFM embeddings generated.")
    return user_embeds, item_embeds


def generate_user_features():
    articles = pd.read_csv("cleaned_data.csv")
    articles = articles.dropna(subset=["title", "abstract"])
    articles["fos"] = articles["fos"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    articles["keywords"] = articles["keywords"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    articles["venue_popularity"] = articles["venue"].map(articles["venue"].value_counts().to_dict())

    interactions = load_user_interactions()
    action_weights = {'LIKE': 2, 'CLICKED': 1, 'COMMENT': 0, 'DISLIKE': 0}
    interactions['label'] = interactions['action'].map(action_weights).astype(float)

    print("🔍 Performing sentiment analysis...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    results = sentiment_pipeline(interactions["comment"].fillna("").tolist(), batch_size=32, truncation=True)

    interactions["sentiment_score"] = [
        res["score"] if res["label"] == "POSITIVE" else -res["score"]
        for res in results
    ]
    for i, row in interactions.iterrows():
        if row["action"] == "COMMENT" and row["sentiment_score"] > 0:
            interactions.at[i, "label"] = max(row["label"], 1.0)

    agg = interactions.groupby(['user_id', 'paper_id']).agg({
        'label': 'max',
        'sentiment_score': 'mean'
    }).reset_index()

    print("💡 Encoding article embeddings...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    articles["content"] = articles["title"] + ". " + articles["abstract"]
    articles["embedding"] = list(model.encode(articles["content"].tolist(), show_progress_bar=True))

    merged = agg.merge(articles, left_on="paper_id", right_on="_id", how="inner")

    user_profiles = {}
    user_fos = defaultdict(set)
    user_keywords = defaultdict(set)

    for user_id, group in merged.groupby("user_id"):
        liked = group[group["label"] >= 1.0]
        user_profiles[user_id] = np.mean(np.stack(liked["embedding"].values), axis=0) if not liked.empty else np.zeros(384)
        for _, row in liked.iterrows():
            user_fos[user_id].update(row["fos"])
            user_keywords[user_id].update(row["keywords"])

    def content_sim(row):
        return cosine_similarity([user_profiles.get(row["user_id"], np.zeros(384))], [row["embedding"]])[0][0]

    def fos_overlap(row):
        a = user_fos.get(row["user_id"], set())
        b = set(row["fos"])
        return len(a & b) / (len(a | b) + 1e-5)

    def keyword_match(row):
        a = user_keywords.get(row["user_id"], set())
        b = set(row["keywords"])
        return len(a & b) / (len(a | b) + 1e-5)

    print("🔧 Computing similarity metrics...")
    merged["content_sim"] = merged.apply(content_sim, axis=1)
    merged["fos_overlap"] = merged.apply(fos_overlap, axis=1)
    merged["keyword_match"] = merged.apply(keyword_match, axis=1)

    features = pd.DataFrame({
        "user_id": merged["user_id"],
        "article_id": merged["_id"],
        "label": merged["label"],
        "sentiment_score": merged["sentiment_score"],
        "n_citation": merged["n_citation"].astype(float),
        "year": merged["year"].astype(int),
        "venue_popularity": merged["venue_popularity"],
        "content_sim": merged["content_sim"],
        "fos_overlap": merged["fos_overlap"],
        "keyword_match": merged["keyword_match"]
    })

    # Matrix Factorization Features
    user_embed_df, item_embed_df = generate_latent_embeddings(interactions)
    features = features.merge(user_embed_df, on="user_id", how="left")
    features = features.merge(item_embed_df, on="article_id", how="left")

    embedding_cols = user_embed_df.columns.difference(["user_id"]).tolist() + item_embed_df.columns.difference(["article_id"]).tolist()
    features[embedding_cols] = features[embedding_cols].fillna(0)

    # Encode IDs
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    features["user_id_enc"] = le_user.fit_transform(features["user_id"])
    features["article_id_enc"] = le_item.fit_transform(features["article_id"])

    os.makedirs("data", exist_ok=True)
    features.to_csv("data/training_features.csv", index=False)
    print("✅ Feature set with LightFM embeddings saved.")


def train_ranking_model():
    data = pd.read_csv("data/training_features.csv")

    # Identify embedding columns
    embedding_cols = [col for col in data.columns if col not in {
        "user_id", "article_id", "user_id_enc", "article_id_enc", "label"
    }]

    X = data[embedding_cols]
    y = data["label"]
    user_ids = data["user_id_enc"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    train_users = user_ids[:len(X_train)]
    val_users = user_ids[len(X_train):]

    train_group = pd.Series(train_users).value_counts().sort_index().tolist()
    val_group = pd.Series(val_users).value_counts().sort_index().tolist()

    dtrain = lgb.Dataset(X_train, label=y_train, group=train_group)
    dval = lgb.Dataset(X_val, label=y_val, group=val_group)

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

    print("🚀 Training LightGBM LambdaRank model...")
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        num_boost_round=300
    )

    os.makedirs("models", exist_ok=True)
    model.save_model("models/ltr_model.txt")
    print("✅ Model saved at models/ltr_model.txt")
