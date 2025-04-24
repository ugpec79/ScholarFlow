import os
import torch
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
from django.core.cache import cache
from .gnn_train import HGTModel, load_graph_from_neo4j
from django.http import JsonResponse

# Constants
LLAMA_API = "http://localhost:11434/api/generate"
EMBEDDING_FILE = "embeddings/paper_embeddings.pt"
ID_MAP_FILE = "embeddings/paper_id_map.pkl"
CACHE_TTL = 3600  # cache timeout in seconds (1 hour)

# Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# Global cache
paper_embeddings, paper_id_map = None, None


def ensure_embeddings():
    global paper_embeddings, paper_id_map

    if os.path.exists(EMBEDDING_FILE) and os.path.exists(ID_MAP_FILE):
        paper_embeddings = torch.load(EMBEDDING_FILE)
        with open(ID_MAP_FILE, "rb") as f:
            paper_id_map = pickle.load(f)
    else:
        data, node_ids = load_graph_from_neo4j()
        for ntype in data.node_types:
            data[ntype].x = torch.randn(data[ntype].num_nodes, 64)

        model = HGTModel(data.metadata(), hidden_channels=64)
        model.load_state_dict(torch.load("contrastive_model.pt"))
        model.eval()

        with torch.no_grad():
            out = model(
                data.collect("x"),
                data.collect("edge_index"),
                data.collect("edge_weight"),
            )
            paper_embeddings = out["Paper"].detach()
            paper_id_map = node_ids["Paper"]

        os.makedirs("embeddings", exist_ok=True)
        torch.save(paper_embeddings, EMBEDDING_FILE)
        with open(ID_MAP_FILE, "wb") as f:
            pickle.dump(paper_id_map, f)

    return paper_embeddings, paper_id_map


def fetch_paper_details(paper_ids):
    papers = {}
    with driver.session() as session:
        query = """
        UNWIND $ids AS pid
        MATCH (p:Paper {_id: pid})
        RETURN properties(p) AS props
        """
        results = session.run(query, ids=paper_ids)
        for record in results:
            props = record["props"]
            papers[props["_id"]] = props
    return papers


def recommend_similar_papers(paper_id, top_k=5):
    if paper_id not in paper_id_map:
        return []

    idx = paper_id_map[paper_id]
    query_embedding = paper_embeddings[idx].unsqueeze(0)
    similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
    top_indices = similarities.argsort()[::-1]

    recommendations = []
    for i in top_indices:
        if i != idx:
            recommended_id = next(k for k, v in paper_id_map.items() if v == i)
            recommendations.append((recommended_id, float(similarities[i])))
            if len(recommendations) >= top_k:
                break
    return recommendations


def generate_llama_explanation(
    input_paper, recommended_paper, similarity_score, common_connections
):
    # (unchanged prompt construction)
    prompt = f"""
You are an academic assistant. Explain why the following research paper was recommended based on both semantic similarity and graph-based relationships like common keywords, fields of study (FoS), venues, authors, and citations.

User's Source Paper:
- Title: {input_paper.get('title', 'N/A')}
- Abstract: {input_paper.get('abstract', 'N/A')}
- Fields: {', '.join(input_paper.get('fos', []))}
- Keywords: {', '.join(input_paper.get('keywords', []))}
- Venue: {input_paper.get('venue', 'N/A')}
- Authors: {', '.join(input_paper.get('authors', []))}
- Year: {input_paper.get('year', 'Unknown')}

Recommended Paper:
- Title: {recommended_paper.get('title', 'N/A')}
- Abstract: {recommended_paper.get('abstract', 'N/A')}
- Fields: {', '.join(recommended_paper.get('fos', []))}
- Keywords: {', '.join(recommended_paper.get('keywords', []))}
- Venue: {recommended_paper.get('venue', 'Unknown')}
- Authors: {', '.join(recommended_paper.get('authors', []))}
- Year: {recommended_paper.get('year', 'Unknown')}
- Citation Count: {recommended_paper.get('n_citation', 'N/A')}
- Semantic Similarity: {round(similarity_score, 4)}

Graph-based Connections:
- Similar Venue: {'Yes' if 'venue' in common_connections else 'No'}
- Common Authors: {', '.join(common_connections.get('authors', [])) if 'authors' in common_connections else 'None'}
- Common Keywords: {', '.join(common_connections.get('keywords', [])) if 'keywords' in common_connections else 'None'}
- Common Fields of Study (FoS): {', '.join(common_connections.get('fos', [])) if 'fos' in common_connections else 'None'}
- Citation Relationship: {'Yes' if 'citation' in common_connections else 'No'}

Write a short, human-readable explanation (1–2 sentences) about why this paper is a good match.
"""
    try:
        response = requests.post(
            LLAMA_API, json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Explanation generation failed: {e}"


def get_recommendations_data(paper_id, top_k=5):
    # Check cache first
    cache_key = f"recommendations:{paper_id}:{top_k}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    ensure_embeddings()
    recommendations = recommend_similar_papers(paper_id, top_k)
    paper_ids = [paper_id] + [pid for pid, _ in recommendations]
    all_papers = fetch_paper_details(paper_ids)

    input_paper = all_papers.get(paper_id, {})
    rec_data = []
    for pid, sim in recommendations:
        rec_paper = all_papers.get(pid, {})
        common_connections = {
            "venue": input_paper.get("venue", "") == rec_paper.get("venue", ""),
            "authors": list(
                set(input_paper.get("authors", [])) & set(rec_paper.get("authors", []))
            ),
            "keywords": list(
                set(input_paper.get("keywords", []))
                & set(rec_paper.get("keywords", []))
            ),
            "fos": list(
                set(input_paper.get("fos", [])) & set(rec_paper.get("fos", []))
            ),
            "citation": input_paper.get("citations", 0) > 0
            and rec_paper.get("citations", 0) > 0,
        }
        explanation = generate_llama_explanation(
            input_paper, rec_paper, sim, common_connections
        )
        rec_data.append(
            {
                "id": pid,
                "similarity": sim,
                "title": rec_paper.get("title", "Unknown"),
                "venue": rec_paper.get("venue", ""),
                "year": rec_paper.get("year", ""),
                "citations": rec_paper.get("n_citation", 0),
                "explanation": explanation,
            }
        )

    # Prepare result
    result = {
        "input_paper": {
            "id": paper_id,
            "title": input_paper.get("title", "Unknown"),
            "year": int(input_paper.get("year", 0))
            if input_paper.get("year")
            else None,
            "n_citation": int(input_paper.get("n_citation", 0))
            if input_paper.get("n_citation")
            else 0,
            "doi": input_paper.get("doi", ""),
            "abstract": input_paper.get("abstract", ""),
            "lang": input_paper.get("lang", ""),
            "url": input_paper.get("url", ""),
            "fos": input_paper.get("fos", []),
            "keywords": input_paper.get("keywords", []),
            "authors": input_paper.get("authors", []),
            "venue": input_paper.get("venue", {}),
            "references": input_paper.get("references", []),
        },
        "recommendations": rec_data,
    }

    # Cache the result
    cache.set(cache_key, result, CACHE_TTL)
    return result


def get_recommendations(request, paper_id):
    response_data = get_recommendations_data(paper_id)
    return JsonResponse(response_data, safe=False)
