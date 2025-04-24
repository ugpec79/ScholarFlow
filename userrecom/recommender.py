import os
import pandas as pd
import numpy as np
import ast
import pickle
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import lightgbm as lgb
from django.conf import settings  # Import settings
from django.core.cache import cache  # Django cache

# --- Constants ---
MODEL_PATH = "models/ltr_model.txt"
DATA_PATH = "cleaned_data.csv"
INTERACTIONS_PATH = "user_actions.csv"
EMBEDDINGS_CACHE = "data/article_embeddings.pkl"
TOP_N = 10
LLAMA_API = settings.LLAMA_API  # Ollama REST API
CACHE_TTL = 3600  # cache duration for recommendations


def generate_llama_explanation(article, user_info):
    prompt = f"""
You are an academic recommendation assistant. Your job is to explain why a research paper was recommended to a user, based on their interests and several scoring features.

User Interests:
- Fields of Study: {', '.join(user_info['fos'])}
- Keywords: {', '.join(user_info['keywords'])}

Recommended Article:
- Title: {article['title']}
- Abstract: {article.get('abstract', 'N/A')}
- Fields of Study: {', '.join(article['fos'])}
- Keywords: {', '.join(article['keywords'])}
- Venue: {article['venue']} (popularity score: {article['venue_popularity']})
- Year: {article['year']}
- Citation Count: {article['n_citation']}

Scoring Features:
- Semantic Similarity: {round(article['content_sim'], 4)}
- Field of Study Overlap: {round(article['fos_overlap'], 4)}
- Keyword Match: {round(article['keyword_match'], 4)}
- Sentiment Score: {round(article['sentiment_score'], 4)}

Using this information, write a short, clear explanation (2-3 sentences) for why this paper was recommended.
"""
    try:
        response = requests.post(
            LLAMA_API, json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        print("LLaMA request failed:", e)
        return "Explanation not available."


def load_or_generate_article_embeddings(force_refresh=False):
    if not force_refresh and os.path.exists(EMBEDDINGS_CACHE):
        print("📦 Loading cached article embeddings...")
        with open(EMBEDDINGS_CACHE, "rb") as f:
            return pickle.load(f)

    print("📡 Generating article embeddings from scratch...")
    articles = pd.read_csv(DATA_PATH)
    articles = articles.dropna(subset=["title", "abstract"])
    articles["fos"] = articles["fos"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    articles["keywords"] = articles["keywords"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    venue_popularity = articles["venue"].value_counts().to_dict()
    articles["venue_popularity"] = articles["venue"].map(
        lambda v: venue_popularity.get(v, 0)
    )

    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    articles["combined_text"] = (
        articles["title"]
        + ". "
        + articles["abstract"]
        + ". "
        + articles["keywords"].apply(lambda kws: " ".join(kws))
        + ". "
        + articles["fos"].apply(lambda fos: " ".join(fos))
    )

    articles["embedding"] = embedding_model.encode(
        articles["combined_text"].tolist(), show_progress_bar=True
    ).tolist()

    os.makedirs("data", exist_ok=True)
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(articles, f)

    return articles


# --- Main Recommendation Function with Caching ---
def get_user_recommendations(user_id, top_n=TOP_N):
    cache_key = f"user_recs:{user_id}:{top_n}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    model = lgb.Booster(model_file=MODEL_PATH)
    articles = load_or_generate_article_embeddings()

    interactions = pd.read_csv(INTERACTIONS_PATH)
    interactions = interactions[interactions["user_id"] == user_id]
    if interactions.empty:
        return []

    sentiment_pipeline = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    interactions["comment"] = interactions["comment"].fillna("")
    interactions["sentiment_score"] = 0.0

    results = sentiment_pipeline(
        interactions["comment"].tolist(), batch_size=32, truncation=True
    )
    for i, res in enumerate(results):
        score = res["score"]
        sentiment = 1.0 if res["label"] == "POSITIVE" else -1.0
        interactions.at[i, "sentiment_score"] = score * sentiment

    weights = {"LIKE": 1.0, "COMMENT": 0.7, "CLICKED": 0.3, "DISLIKE": 0.0}
    interactions["weight"] = interactions["action"].map(weights)
    interactions = interactions[interactions["weight"] > 0]
    interactions = interactions.merge(
        articles, left_on="paper_id", right_on="_id", how="inner"
    )

    embed_acc = []
    fos_set = set()
    keyword_set = set()

    for _, row in interactions.iterrows():
        embed = np.array(row["embedding"])
        weight = row["weight"]
        embed_acc.append(embed * weight)
        fos_set.update(row["fos"])
        keyword_set.update(row["keywords"])

    user_embed = np.sum(embed_acc, axis=0) / np.sum(interactions["weight"])
    seen_ids = set(interactions["paper_id"])
    candidates = articles[~articles["_id"].isin(seen_ids)].copy()

    def compute_content_sim(row):
        return cosine_similarity([user_embed], [row["embedding"]])[0][0]

    def compute_fos_overlap(row):
        return len(fos_set.intersection(set(row["fos"]))) / (
            len(fos_set.union(set(row["fos"]))) + 1e-5
        )

    def compute_keyword_match(row):
        return len(keyword_set.intersection(set(row["keywords"]))) / (
            len(keyword_set.union(set(row["keywords"]))) + 1e-5
        )

    candidates["content_sim"] = candidates.apply(compute_content_sim, axis=1)
    candidates["fos_overlap"] = candidates.apply(compute_fos_overlap, axis=1)
    candidates["keyword_match"] = candidates.apply(compute_keyword_match, axis=1)
    candidates["sentiment_score"] = 0

    feature_cols = [
        "sentiment_score",
        "n_citation",
        "year",
        "venue_popularity",
        "content_sim",
        "fos_overlap",
        "keyword_match",
    ]
    candidates = candidates.dropna(subset=feature_cols)
    X_test = candidates[feature_cols]
    candidates["score"] = model.predict(X_test)

    top_n_results = candidates.sort_values("score", ascending=False).head(top_n)
    recommendations = []
    user_info = {"fos": list(fos_set), "keywords": list(keyword_set)}

    for _, row in top_n_results.iterrows():
        explanation = generate_llama_explanation(row, user_info)
        recommendations.append(
            {
                "title": row["title"],
                "year": int(row["year"]),
                "venue": row["venue"],
                "score": round(row["score"], 4),
                "fos": row["fos"],
                "keywords": row["keywords"][:10],
                "explanation": explanation,
            }
        )

    cache.set(cache_key, recommendations, CACHE_TTL)
    return recommendations
