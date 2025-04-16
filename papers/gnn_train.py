# gnn_train.py
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from neo4j import GraphDatabase
import numpy as np
import random
import pickle
import os

# ---- Reproducibility ----
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed()

# ---- Neo4j Driver ----
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# ---- Edge Importance ----
EDGE_IMPORTANCE = {
    ("Paper", "CITES", "Paper"): 2.0,
    ("Paper", "HAS_TOPIC", "FOS"): 1.5,
    ("Paper", "HAS_KEYWORD", "Keyword"): 1.5,
    ("Paper", "PUBLISHED_IN", "Venue"): 1.2,
    ("Paper", "AUTHORED_BY", "Author"): 1.2
}

# ---- Graph Loading ----
def load_graph_from_neo4j():
    data = HeteroData()
    node_ids = {ntype: {} for ntype in ['Paper', 'Keyword', 'FOS', 'Venue', 'Author']}
    counters = {k: 0 for k in node_ids}

    with driver.session() as session:
        for ntype in node_ids:
            key = 'name' if ntype in ['Keyword', 'FOS'] else '_id'
            query = f"MATCH (n:{ntype}) RETURN DISTINCT n.{key} AS id"
            for record in session.run(query):
                node_ids[ntype][record['id']] = counters[ntype]
                counters[ntype] += 1
            data[ntype].num_nodes = counters[ntype]

        edge_types = [
            ("Paper", "HAS_KEYWORD", "Keyword"),
            ("Keyword", "KEYWORD_OF", "Paper"),
            ("Paper", "HAS_TOPIC", "FOS"),
            ("FOS", "TOPIC_OF", "Paper"),
            ("Paper", "PUBLISHED_IN", "Venue"),
            ("Venue", "HOSTS", "Paper"),
            ("Paper", "AUTHORED_BY", "Author"),
            ("Author", "AUTHORED", "Paper"),
            ("Paper", "CITES", "Paper"),
            ("Paper", "CITED_BY", "Paper")
        ]

        for src, rel, dst in edge_types:
            src_ids, dst_ids = [], []
            src_key = 'name' if src in ['Keyword', 'FOS'] else '_id'
            dst_key = 'name' if dst in ['Keyword', 'FOS'] else '_id'

            query = f"""
                MATCH (a:{src})-[:{rel}]->(b:{dst})
                RETURN a.{src_key} AS a, b.{dst_key} AS b
            """
            for record in session.run(query):
                a, b = record['a'], record['b']
                if a in node_ids[src] and b in node_ids[dst]:
                    src_ids.append(node_ids[src][a])
                    dst_ids.append(node_ids[dst][b])

            if src_ids:
                edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
                data[(src, rel, dst)].edge_index = edge_index
                weight = EDGE_IMPORTANCE.get((src, rel, dst), 1.0)
                data[(src, rel, dst)].edge_weight = torch.full((edge_index.size(1),), weight)

    return data, node_ids

# ---- Pair Generation ----
def build_positive_pairs(session, node_ids):
    pairs = set()

    def add_pairs(query, key_a, key_b):
        for record in session.run(query):
            a, b = record[key_a], record[key_b]
            if a in node_ids["Paper"] and b in node_ids["Paper"]:
                aid, bid = node_ids["Paper"][a], node_ids["Paper"][b]
                if aid != bid:
                    pairs.add((aid, bid))

    add_pairs("MATCH (a:Paper)-[:CITES]->(b:Paper) RETURN a._id AS a, b._id AS b", "a", "b")
    add_pairs("""
        MATCH (p1:Paper)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(p2:Paper)
        RETURN DISTINCT p1._id AS a, p2._id AS b
    """, "a", "b")
    add_pairs("""
        MATCH (p1:Paper)-[:HAS_TOPIC]->(f:FOS)<-[:HAS_TOPIC]-(p2:Paper)
        RETURN DISTINCT p1._id AS a, p2._id AS b
    """, "a", "b")
    add_pairs("""
        MATCH (p1:Paper)-[:PUBLISHED_IN]->(v:Venue)<-[:PUBLISHED_IN]-(p2:Paper)
        RETURN DISTINCT p1._id AS a, p2._id AS b
    """, "a", "b")
    add_pairs("""
        MATCH (p1:Paper)-[:AUTHORED_BY]->(a:Author)<-[:AUTHORED_BY]-(p2:Paper)
        RETURN DISTINCT p1._id AS a, p2._id AS b
    """, "a", "b")

    return list(pairs)

def build_negative_pairs(num_nodes, positive_pairs, num_samples=10000):
    mask = np.zeros((num_nodes, num_nodes), dtype=bool)
    for a, b in positive_pairs:
        mask[a, b] = True
        mask[b, a] = True

    neg_pairs = []
    attempts = 0
    max_attempts = num_samples * 10

    while len(neg_pairs) < num_samples and attempts < max_attempts:
        a = np.random.randint(0, num_nodes)
        b = np.random.randint(0, num_nodes)
        if a != b and not mask[a, b]:
            neg_pairs.append((a, b))
            mask[a, b] = True
            mask[b, a] = True
        attempts += 1

    return neg_pairs

# ---- Model ----
class WeightedHGTConv(HGTConv):
    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        out = super().forward(x_dict, edge_index_dict)
        if edge_weight_dict:
            for ntype in out:
                for (src, rel, dst), weight in edge_weight_dict.items():
                    if isinstance(weight, (float, int)):
                        out[ntype] *= weight
        return out

class HGTModel(Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=64, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            WeightedHGTConv(
                in_channels={nt: hidden_channels for nt in metadata[0]},
                out_channels=hidden_channels,
                metadata=metadata,
                heads=2
            ) for _ in range(num_layers)
        ])
        self.lin_dict = torch.nn.ModuleDict({
            ntype: Linear(hidden_channels, out_channels) for ntype in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict)
        return {k: self.lin_dict[k](v) for k, v in x_dict.items()}

class ContrastiveLoss(Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, label):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.sum(z1 * z2, dim=1) / self.temperature
        return F.binary_cross_entropy_with_logits(sim, label.float())

# ---- Training ----
def train_model(epochs=100):
    data, node_ids = load_graph_from_neo4j()
    hidden_channels = 64
    for ntype in data.node_types:
        data[ntype].x = torch.randn(data[ntype].num_nodes, hidden_channels)

    model = HGTModel(data.metadata(), hidden_channels=hidden_channels)
    loss_fn = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    with driver.session() as session:
        pos_pairs = build_positive_pairs(session, node_ids)
        neg_pairs = build_negative_pairs(data["Paper"].num_nodes, pos_pairs, len(pos_pairs))
        all_pairs = pos_pairs + neg_pairs
        labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.collect('x'), data.collect('edge_index'), data.collect('edge_weight'))

        z = out["Paper"]
        z1 = torch.stack([z[a] for a, b in all_pairs])
        z2 = torch.stack([z[b] for a, b in all_pairs])
        label_tensor = torch.tensor(labels)

        loss = loss_fn(z1, z2, label_tensor)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    os.makedirs("embeddings", exist_ok=True)
    torch.save(model.state_dict(), "contrastive_model.pt")
    with open("embeddings/contrastive_embeddings.pkl", "wb") as f:
        pickle.dump((out['Paper'].detach(), node_ids['Paper']), f)
