# bootstrap/apps.py

from django.apps import AppConfig
import os


class BootstrapConfig(AppConfig):
    name = "bootstrap"

    def ready(self):
        from .utils import start_scheduler

        # Start the periodic scheduler
        if os.environ.get("RUN_MAIN") == "true":
            start_scheduler()
