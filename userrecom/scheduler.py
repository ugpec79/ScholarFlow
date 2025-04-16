from apscheduler.schedulers.background import BackgroundScheduler
from .retrain import retrain_model_pipeline

scheduler = BackgroundScheduler()

def start_scheduler():
    if not scheduler.running:
        scheduler.add_job(retrain_model_pipeline, 'interval', days=1)
        scheduler.start()
