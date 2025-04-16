import uuid
from django.db import models

class GuestUser(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    user_agent = models.CharField(max_length=512)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.uuid)