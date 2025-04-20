# trendrecom/utils.py

import pandas as pd
from datetime import datetime
from math import exp, log
from transformers import pipeline
import os


def generate_trending_papers():
    # --- Step 1: Load Paper Data ---
    papers = pd.read_csv("cleaned_data.csv")
    papers["publish_date"] = pd.to_datetime(
        papers["year"].astype(str) + "-01-01", errors="coerce"
    )
    papers.rename(
        columns={"_id": "paper_id", "n_citation": "citation_count"}, inplace=True
    )
    papers["citation_count"] = papers["citation_count"].fillna(0)

    # --- Step 2: Load User Actions (if exists) ---
    actions_path = "user_actions.csv"
    has_user_data = os.path.exists(actions_path) and os.path.getsize(actions_path) > 0

    if has_user_data:
        actions = pd.read_csv(actions_path)
        required_cols = {"paper_id", "action", "comment"}
        if not required_cols.issubset(actions.columns):
            print(
                "⚠️ user_actions.csv is missing required columns. Skipping interaction analysis."
            )
            has_user_data = False
    else:
        print(
            "ℹ️ No user_actions.csv found or file is empty. Proceeding without user interactions."
        )
        actions = pd.DataFrame()

    # --- Step 3: Batched Sentiment Analysis (if possible) ---
    if has_user_data:
        print("🔍 Performing batched sentiment analysis using DistilBERT...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

        comments = actions[actions["action"] == "COMMENT"].copy()
        comments["comment"] = comments["comment"].apply(
            lambda x: x if isinstance(x, str) else ""
        )
        comments = comments[comments["comment"].str.strip() != ""]

        # Batched analysis
        sentiment_scores = []
        batch_size = 1000
        comment_list = comments["comment"].tolist()
        for i in range(0, len(comment_list), batch_size):
            batch = comment_list[i : i + batch_size]
            results = sentiment_pipeline(batch)
            for res in results:
                score = res["score"] if res["label"] == "POSITIVE" else -res["score"]
                sentiment_scores.append(score)
        comments["sentiment_score"] = sentiment_scores

        # Aggregation
        interaction_counts = (
            actions.groupby(["paper_id", "action"]).size().unstack(fill_value=0)
        )
        sentiment_summary = (
            comments.groupby("paper_id")["sentiment_score"]
            .mean()
            .reset_index()
            .rename(columns={"sentiment_score": "avg_sentiment"})
        )
        comment_counts = (
            comments.groupby("paper_id").size().reset_index(name="comment_count")
        )
    else:
        interaction_counts = pd.DataFrame()
        sentiment_summary = pd.DataFrame(columns=["paper_id", "avg_sentiment"])
        comment_counts = pd.DataFrame(columns=["paper_id", "comment_count"])

    # --- Step 4: Merge All Data ---
    paper_data = papers.copy()

    # Ensure necessary interaction columns are present before merge
    if not interaction_counts.empty:
        for col in ["LIKE", "DISLIKE", "CLICKED"]:
            if col not in interaction_counts.columns:
                interaction_counts[col] = 0
        paper_data = paper_data.merge(interaction_counts, on="paper_id", how="left")

    if not sentiment_summary.empty:
        paper_data = paper_data.merge(sentiment_summary, on="paper_id", how="left")

    if not comment_counts.empty:
        paper_data = paper_data.merge(comment_counts, on="paper_id", how="left")

    # Fill missing feedback fields in bulk
    for col in ["LIKE", "DISLIKE", "CLICKED", "comment_count", "avg_sentiment"]:
        if col not in paper_data.columns:
            paper_data[col] = 0
    paper_data[["LIKE", "DISLIKE", "CLICKED", "comment_count", "avg_sentiment"]] = (
        paper_data[
            ["LIKE", "DISLIKE", "CLICKED", "comment_count", "avg_sentiment"]
        ].fillna(0)
    )

    # --- Step 5: Compute Raw Feedback & Citation Scores ---
    paper_data["raw_citation_score"] = paper_data["citation_count"].apply(
        lambda x: log(x + 1)
    )
    paper_data["raw_feedback_score"] = (
        paper_data["LIKE"] * 1.5
        + paper_data["CLICKED"] * 1.0
        + paper_data["comment_count"] * 1.0
        + paper_data["avg_sentiment"] * paper_data["comment_count"]
        - paper_data["DISLIKE"] * 1.2
    )

    # Normalize scores to 0–1
    citation_min = paper_data["raw_citation_score"].min()
    citation_max = paper_data["raw_citation_score"].max()
    paper_data["citation_score"] = (paper_data["raw_citation_score"] - citation_min) / (
        citation_max - citation_min + 1e-9
    )

    feedback_min = paper_data["raw_feedback_score"].min()
    feedback_max = paper_data["raw_feedback_score"].max()
    paper_data["feedback_score"] = (paper_data["raw_feedback_score"] - feedback_min) / (
        feedback_max - feedback_min + 1e-9
    )

    # --- Step 6: Vectorized Final Scoring ---
    years_old = (datetime.now().year - paper_data["publish_date"].dt.year).fillna(0) + 1
    paper_data["recency_weight"] = (-0.3 * years_old).apply(exp)

    # --- Step 6: Adjusted Final Scoring with Weighted Feedback ---
    feedback_weight = 0.7
    citation_weight = 0.3
    paper_data["trending_score"] = (
        (
            paper_data["feedback_score"] * feedback_weight
            + paper_data["citation_score"] * citation_weight
        )
        * paper_data["recency_weight"]
    ).round(4)

    # --- Step 7: Sort + Export ---
    top_trending = paper_data.sort_values(by="trending_score", ascending=False)
    top_trending[["paper_id", "title", "trending_score"]].to_csv(
        "trending_papers.csv", index=False
    )

    # --- Step 8: Display ---
    print("✅ Trending recommendations saved to 'trending_papers.csv'\n")
