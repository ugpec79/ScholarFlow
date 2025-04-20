def run_indexing():
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from elasticsearch import Elasticsearch, helpers
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from tqdm import tqdm
    import ast
    from uuid import UUID
    import traceback
    from django.conf import settings  # Import settings

    CSV_FILE = "cleaned_data.csv"
    ES_URL = settings.ES_URL
    QDRANT_URL = settings.QDRANT_URL
    QDRANT_PORT = settings.QDRANT_PORT
    ES_INDEX_NAME = "research_papers"
    QDRANT_COLLECTION_NAME = "research_papers"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    BULK_SIZE = 100

    print("📄 Loading CSV...")
    df = pd.read_csv(CSV_FILE)
    df = df.head(1000)

    print("⚙️ Connecting to Elasticsearch and Qdrant...")
    es = Elasticsearch(ES_URL)
    qdrant = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

    # Skip indexing if both exist
    if es.indices.exists(index=ES_INDEX_NAME) and QDRANT_COLLECTION_NAME in [
        c.name for c in qdrant.get_collections().collections
    ]:
        print("✅ Index and collection already exist. Skipping indexing.")
        return

    if es.indices.exists(index=ES_INDEX_NAME):
        print(f"🗑️ Deleting Elasticsearch index: {ES_INDEX_NAME}")
        es.indices.delete(index=ES_INDEX_NAME)

    if QDRANT_COLLECTION_NAME in [c.name for c in qdrant.get_collections().collections]:
        print(f"🗑️ Deleting Qdrant collection: {QDRANT_COLLECTION_NAME}")
        qdrant.delete_collection(QDRANT_COLLECTION_NAME)

    es.indices.create(
        index=ES_INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "abstract": {"type": "text"},
                    "year": {"type": "integer"},
                    "n_citation": {"type": "integer"},
                    "doi": {"type": "keyword"},
                    "lang": {"type": "keyword"},
                    "keywords": {"type": "keyword"},
                    "fos": {"type": "keyword"},
                    "authors": {"type": "nested"},
                    "venue": {"type": "object"},
                    "references": {"type": "keyword"},
                    "urls": {"type": "keyword"},
                }
            }
        },
    )

    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    print("🔍 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    def parse_paper_row(paper):
        paper_id = paper["_id"]
        title = paper.get("title", "")
        year = int(paper.get("year", 0)) if paper.get("year") else None
        n_citation = int(paper.get("n_citation", 0)) if paper.get("n_citation") else 0
        doi = paper.get("doi", "")
        abstract = paper.get("abstract", "")
        lang = paper.get("lang", "")
        urls = paper.get("dblp_url", "")
        try:
            fos = [
                f.strip().lower() for f in ast.literal_eval(paper.get("fos", "[]")) if f
            ]
        except:
            fos = []
        try:
            keywords = [
                k.strip().lower()
                for k in ast.literal_eval(paper.get("keywords", "[]"))
                if k
            ]
        except:
            keywords = []
        try:
            authors = ast.literal_eval(paper.get("authors", "[]"))
        except:
            authors = []
        try:
            venue = ast.literal_eval(paper.get("venue", "{}"))
        except:
            venue = {}
        try:
            references = ast.literal_eval(paper.get("references", "[]"))
        except:
            references = []

        return {
            "_id": paper_id,
            "title": title,
            "year": year,
            "n_citation": n_citation,
            "doi": doi,
            "abstract": abstract,
            "lang": lang,
            "fos": fos,
            "keywords": keywords,
            "authors": authors,
            "venue": venue,
            "references": references,
            "urls": urls,
        }

    def get_point_id(paper_id):
        try:
            UUID(paper_id)
            return paper_id
        except Exception:
            return abs(hash(paper_id)) % (10**9)

    print("🚀 Indexing documents...")
    es_docs = []
    qdrant_points = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            paper = parse_paper_row(row)
            embedding_text = f"{paper['title']} {paper['abstract']} {' '.join(paper['keywords'])} {' '.join(paper['fos'])}"
            embedding = model.encode(embedding_text).tolist()

            es_docs.append(
                {
                    "_index": ES_INDEX_NAME,
                    "_id": paper["_id"],
                    "_source": {
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "year": paper["year"],
                        "n_citation": paper["n_citation"],
                        "doi": paper["doi"],
                        "lang": paper["lang"],
                        "keywords": paper["keywords"],
                        "fos": paper["fos"],
                        "authors": paper["authors"],
                        "venue": paper["venue"],
                        "references": paper["references"],
                        "urls": paper["urls"],
                    },
                }
            )

            qdrant_points.append(
                models.PointStruct(
                    id=get_point_id(paper["_id"]), vector=embedding, payload=paper
                )
            )

            if len(es_docs) >= BULK_SIZE:
                helpers.bulk(es, es_docs)
                es_docs = []
            if len(qdrant_points) >= BULK_SIZE:
                qdrant.upsert(QDRANT_COLLECTION_NAME, points=qdrant_points)
                qdrant_points = []

        except Exception as e:
            print(f"❌ Error at row {idx}: {e}")
            traceback.print_exc()

    if es_docs:
        helpers.bulk(es, es_docs)
    if qdrant_points:
        qdrant.upsert(QDRANT_COLLECTION_NAME, points=qdrant_points)

    print("✅ Indexing complete.")
