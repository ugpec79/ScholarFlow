from django.apps import AppConfig
import threading

class TrendrecomConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'trendrecom'

    def ready(self):
        from .scheduler import start_scheduler

        # Start the periodic scheduler
        start_scheduler()