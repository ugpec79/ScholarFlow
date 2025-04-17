# graph.py
import csv
import json
from neo4j import GraphDatabase
import ast
from tqdm import tqdm
from django.conf import settings  # Import settings
# Neo4j connection config
NEO4J_URI = settings.NEO4J_URI
NEO4J_USER = settings.NEO4J_USER
NEO4J_PASSWORD = settings.NEO4J_PASSWORD
CSV_FILE = "cleaned_data.csv"
NUM_ROWS = 1000
from django.conf import settings  # Import settings
driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_graph():
    # Load first 1000 rows
    papers = []
    all_paper_ids = set()

    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= NUM_ROWS:
                break
            papers.append(row)
            all_paper_ids.add(row["_id"])

    # Clear existing graph
    with driver.session() as session:
        session.run("MATCH ()-[r]->() DELETE r")
        session.run("MATCH (n) DETACH DELETE n")

    # Create/update papers and their relationships (excluding citations)
    def create_or_update_paper(tx, paper):
        try:
            paper_id = paper["_id"]
            title = paper.get("title", "")
            year = int(paper.get("year", 0)) if paper.get("year") else None
            n_citation = int(paper.get("n_citation", 0)) if paper.get("n_citation") else 0
            doi = paper.get("doi", "")
            abstract = paper.get("abstract", "")
            lang = paper.get("lang", "")
            urls = paper.get("dblp_url", "")
            
            # Parse fields
            fos_list = [f.strip().lower() for f in ast.literal_eval(paper.get("fos", "[]")) if f]
            keyword_list = [k.strip().lower() for k in ast.literal_eval(paper.get("keywords", "[]")) if k]
            authors = ast.literal_eval(paper.get("authors", "[]"))
            venue = ast.literal_eval(paper.get("venue", "{}"))

            tx.run("""
                MERGE (p:Paper {_id: $id})
                SET p.title = $title,
                    p.year = $year,
                    p.n_citation = $n_citation,
                    p.doi = $doi,
                    p.abstract = $abstract,
                    p.lang = $lang,
                    p.url = $urls
            """, id=paper_id, title=title, year=year, n_citation=n_citation,
                doi=doi, abstract=abstract, lang=lang, urls=urls)


            for author in authors:
                if "_id" in author and "name" in author:
                    tx.run("""
                        MERGE (a:Author {_id: $aid})
                        SET a.name = $name
                        WITH a
                        MATCH (p:Paper {_id: $pid})
                        MERGE (a)-[:AUTHORED]->(p)
                        MERGE (p)-[:AUTHORED_BY]->(a)
                    """, aid=author["_id"], name=author["name"], pid=paper_id)

            if venue and "_id" in venue:
                tx.run("""
                    MERGE (v:Venue {_id: $vid})
                    SET v.name = $vname, v.raw = $vraw
                    WITH v
                    MATCH (p:Paper {_id: $pid})
                    MERGE (p)-[:PUBLISHED_IN]->(v)
                    MERGE (v)-[:HOSTS]->(p)
                """, vid=venue["_id"], vname=venue.get("name"), vraw=venue.get("raw"), pid=paper_id)


            for fos in fos_list:
                tx.run("""
                    MERGE (f:FOS {name: $name})
                    WITH f
                    MATCH (p:Paper {_id: $pid})
                    MERGE (p)-[:HAS_TOPIC]->(f)
                    MERGE (f)-[:TOPIC_OF]->(p)
                """, name=fos, pid=paper_id)


            for kw in keyword_list:
                tx.run("""
                    MERGE (k:Keyword {name: $name})
                    WITH k
                    MATCH (p:Paper {_id: $pid})
                    MERGE (p)-[:HAS_KEYWORD]->(k)
                    MERGE (k)-[:KEYWORD_OF]->(p)
                """, name=kw, pid=paper_id)


        except Exception as e:
            print(f"❌ Error processing paper {paper.get('_id')}: {e}")

    # Create citation edges (only if both papers exist)
    def create_citation_relation(tx, paper_id, cited_id):
        tx.run("""
            MATCH (p:Paper {_id: $paper_id})
            MATCH (cited:Paper {_id: $cited_id})
            MERGE (p)-[:CITES]->(cited)
            MERGE (cited)-[:CITED_BY]->(p)
        """, paper_id=paper_id, cited_id=cited_id)

    # Pass 1: Create all papers and metadata
    with driver.session() as session:
        for row in tqdm(papers, desc="🚀 Creating paper nodes"):
            session.execute_write(create_or_update_paper, row)

    # Pass 2: Add CITES relationships
    with driver.session() as session:
        for row in tqdm(papers, desc="🔗 Creating citation links"):
            try:
                references = ast.literal_eval(row.get("references", "[]"))
                for cited_id in references:
                    if cited_id in all_paper_ids:
                        session.execute_write(create_citation_relation, row["_id"], cited_id)
            except Exception:
                continue

    driver.close()
    print("✅ Imported first 1000 papers and all valid internal citations.")

