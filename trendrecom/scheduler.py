from apscheduler.schedulers.background import BackgroundScheduler
from .utils import generate_trending_papers

scheduler = BackgroundScheduler()

def start_scheduler():
    if not scheduler.running:
        scheduler.add_job(generate_trending_papers, 'interval', days=1)
        scheduler.start()
