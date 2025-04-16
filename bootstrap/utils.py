# core/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from userrecom.retrain import retrain_model_pipeline
from trendrecom.utils import generate_trending_papers

scheduler = BackgroundScheduler()

def retrain_and_generate():
    print("Running retrain_model_pipeline...")
    retrain_model_pipeline()
    
    print("Running generate_trending_papers...")
    generate_trending_papers()

def start_scheduler():
    if not scheduler.running:
        scheduler.add_job(retrain_and_generate, 'interval', days=1, id='combined_job')
        scheduler.start()
