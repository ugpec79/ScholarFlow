# bootstrap/management/commands/bootstrap_all.py

from django.core.management.base import BaseCommand
import os
from django.conf import settings

from papers import gnn_train
from query import indexing
from trendrecom.utils import generate_trending_papers

from userrecom.engine import generate_user_features, train_ranking_model


class Command(BaseCommand):
    help = "Runs all bootstrapping tasks before server starts."

    def handle(self, *args, **options):
        try:
            print("🚀 Loading graph data into Neo4j...")
            # graph.run_graph()

            model_path = os.path.join(settings.BASE_DIR, "contrastive_model.pt")
            if not os.path.exists(model_path):
                print("🚀 Training GNN model...")
                gnn_train.train_model()
            else:
                print("✅ GNN model already trained. Skipping training.")

            print("🚀 Running indexing...")
            indexing.run_indexing()
            generate_trending_papers()
            # Create data and models directory if they don't exist
            os.makedirs("data", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            generate_user_features()
            train_ranking_model()

            print("✅ Bootstrapping completed successfully.")

        except Exception as e:
            self.stderr.write(f"❌ Bootstrap error: {e}")
            raise SystemExit(1)
