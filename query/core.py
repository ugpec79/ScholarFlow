from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import redis
import requests
from django.conf import settings  # Import settings

# --- Configuration ---
ES_URL = settings.ES_URL
QDRANT_URL = settings.QDRANT_URL
QDRANT_PORT = settings.QDRANT_PORT
REDIS_HOST = settings.REDIS_HOST
REDIS_PORT = settings.REDIS_PORT

ES_INDEX_NAME = "research_papers"
QDRANT_COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_API = settings.LLAMA_API  # Ollama REST API URL

# --- Connect to Services ---
es = Elasticsearch(ES_URL, request_timeout=30)
qdrant = QdrantClient(QDRANT_URL, port=QDRANT_PORT)
rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
model = SentenceTransformer(EMBEDDING_MODEL)


# --- Redis Query Tracker ---
def add_query_to_redis(user_id, query, ttl=7200):
    rds.lpush(user_id, query)
    rds.expire(user_id, ttl)


def get_past_queries(user_id):
    return rds.lrange(user_id, 0, -1)


# --- Normalize Scores ---
def normalize_score(score, min_score, max_score):
    if max_score == min_score:
        return 0.5
    return (score - min_score) / (max_score - min_score)


# --- Fetch Explanation from Ollama LLaMA ---
def get_explanation_for_result(
    paper_id, current_query, past_query, reason_for_recommendation
):
    prompt = f"Paper ID: {paper_id}\nCurrent Query: {current_query}\nPast Query: {past_query}\nReason for Recommendation: {reason_for_recommendation}\nPlease explain why this paper was recommended."
    payload = {
        "model": "llama3.2",  # Specify the LLaMA model you're using
        "prompt": prompt,
        "max_tokens": 150,  # You can adjust the number of tokens as needed
    }

    headers = {"Content-Type": "application/json"}

    # Send request to Ollama API
    response = requests.post(LLAMA_API, json=payload, headers=headers)

    if response.status_code == 200:
        explanation = response.json().get("response", "")
        return explanation
    else:
        return f"Error generating explanation: {response.status_code}"


# !
# --- Hybrid Search (Elasticsearch + Qdrant) ---
def hybrid_search(query, top_k=10, w1=0.5, w2=0.5):
    es_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title", "abstract", "venue", "authors", "keywords", "fos"],
            }
        }
    }
    es_results = es.search(index=ES_INDEX_NAME, body=es_query, size=top_k)

    query_vector = model.encode(query).tolist()
    qdrant_results = qdrant.search(
        collection_name=QDRANT_COLLECTION_NAME, query_vector=query_vector, limit=top_k
    )

    results = {}
    es_scores = [hit["_score"] for hit in es_results["hits"]["hits"]]
    es_min, es_max = min(es_scores, default=0), max(es_scores, default=1)

    for hit in es_results["hits"]["hits"]:
        paper_id = hit["_id"]
        normalized_es_score = normalize_score(hit["_score"], es_min, es_max)
        results[paper_id] = {
            "_id": paper_id,
            **hit["_source"],
            "es_score": normalized_es_score,
            "qdrant_score": 0,
            "combined_score": w1 * normalized_es_score,
        }

    qdrant_scores = [hit.score for hit in qdrant_results]
    qdrant_min, qdrant_max = (
        min(qdrant_scores, default=0),
        max(qdrant_scores, default=1),
    )

    for hit in qdrant_results:
        paper_id = hit.payload["_id"]
        normalized_qdrant_score = normalize_score(hit.score, qdrant_min, qdrant_max)
        if paper_id in results:
            results[paper_id]["qdrant_score"] = normalized_qdrant_score
            results[paper_id]["combined_score"] = (
                w1 * results[paper_id]["es_score"] + w2 * normalized_qdrant_score
            )
        else:
            results[paper_id] = {
                "_id": paper_id,
                **hit.payload,
                "es_score": 0,
                "qdrant_score": normalized_qdrant_score,
                "combined_score": w2 * normalized_qdrant_score,
            }

    return sorted(
        results.values(),
        key=lambda x: (x["combined_score"], x.get("n_citation", 0)),
        reverse=True,
    )[:top_k]


# --- Fetch Past Query Related Results ---
def fetch_similar_results_with_scores(
    user_id, current_query, similarity_threshold=0.5, top_k=10
):
    past_queries = [q.decode() for q in get_past_queries(user_id)]
    current_embedding = model.encode(current_query).reshape(1, -1)

    combined_results = []
    seen_ids = set()

    for q in past_queries:
        q_embedding = model.encode(q).reshape(1, -1)
        sim_score = cosine_similarity(current_embedding, q_embedding)[0][0]
        if sim_score >= similarity_threshold:
            results = hybrid_search(q, top_k=top_k)
            for res in results:
                if res["_id"] not in seen_ids:
                    res["query_similarity"] = sim_score
                    seen_ids.add(res["_id"])
                    combined_results.append(res)

    return combined_results


# --- Re-rank Results ---
def rerank_combined_results(
    current_results, past_results, current_query, alpha=0.7, beta=0.3, top_k=10
):
    current_embedding = model.encode(current_query).reshape(1, -1)

    for res in current_results:
        res["query_similarity"] = 1.0

    all_results = {res["_id"]: res for res in past_results}
    for res in current_results:
        all_results[res["_id"]] = res

    reranked = []
    for res in all_results.values():
        sim_score = res.get("query_similarity", 0.0)
        final_score = alpha * res["combined_score"] + beta * sim_score
        res["final_score"] = final_score

        # Generate explanation using Ollama LLaMA for each recommended result
        explanation = get_explanation_for_result(
            res["_id"],
            current_query,
            res.get("query_similarity", ""),
            "similar to past query",
        )
        res["explanation"] = explanation

        reranked.append(res)

    return sorted(reranked, key=lambda x: x["final_score"], reverse=True)[:top_k]
