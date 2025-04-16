from django.apps import AppConfig


class UserrecomConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'userrecom'
    def ready(self):
        from .scheduler import start_scheduler

        # Start the periodic scheduler
        start_scheduler()
